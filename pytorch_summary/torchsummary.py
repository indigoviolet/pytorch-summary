from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from functools import reduce
from operator import mul
from pprint import pformat
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
)

import attr
import tabulate
import torch
from humanfriendly import format_number, format_size  # type: ignore
from shapely import Shape, shape
from torch import Tensor, nn
from tqdm import tqdm  # type: ignore

SHAPE_COL_WIDTH = 50
tabulate.PRESERVE_WHITESPACE = True


def mod_isinstance(
    obj: Any, types: Union[Type[nn.Module], Tuple[Type[nn.Module], ...]]
) -> bool:
    """Similar to isinstance(obj, (...)); but it only checks that name of
    the obj ends with the module's name.

    For ex. mod_isinstance(FixedBatchNorm2d(), nn.BatchNorm2d) == True
    """
    if not isinstance(types, tuple):
        types = (types,)

    return any(obj.__class__.__name__.endswith(c.__name__) for c in types)


def prod(x: Iterable) -> float:
    return reduce(mul, x, 1.0)


def make_random_input(
    shape: Tuple[int, ...],
    dtype=torch.FloatTensor,
    device: torch.device = torch.device("cpu"),
    bs: Optional[int] = 2,
):
    """
    Chooses bs=2 by default for batchnorm. bs=None means to squeeze out that dimension
    """
    assert len(shape) == 3, "Expected shape to be (C, H, W), got {shape=}"
    shape = (bs, *shape) if bs is not None else shape
    return torch.rand(*shape).type(dtype).to(device=device)


@attr.s(auto_attribs=True, eq=False)
class ModuleInfo:
    name: str
    module: nn.Module
    input_shape: Shape

    # input_order and output_order will be different in the case of
    # nested modules: the "wrapper" module will appear first in input
    # order, but last in output order. Similarly the innermost module
    # will appear first in output_order.
    input_order: int

    output_shape: Optional[Shape] = None
    output_order: Optional[int] = None

    module_class: str = attr.ib(init=False)
    num_params: int = attr.ib(init=False, default=0)
    trainable: bool = attr.ib(init=False, default=False)

    kernel_size: Optional[int] = attr.ib(init=False, default=None)
    padding: Optional[int] = attr.ib(init=False, default=None)
    stride: Optional[int] = attr.ib(init=False, default=None)

    conv_info: ConvInfo = attr.ib(
        init=False, factory=lambda: ConvInfo(start=0.5, jump=1, receptive_field=1)
    )

    def __attrs_post_init__(self):
        self.module_class = self.module.__class__.__name__

        if hasattr(self.module, "weight") and hasattr(self.module.weight, "size"):
            self.num_params += prod(self.module.weight.size())
            self.trainable = self.module.weight.requires_grad

        if hasattr(self.module, "bias") and hasattr(self.module.bias, "size"):
            self.num_params += prod(self.module.bias.size())

        if mod_isinstance(self.module, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d)):
            self.kernel_size = self._assert_square(self.module, "kernel_size")
            self.stride = self._assert_square(self.module, "stride")
            self.padding = self._assert_square(self.module, "padding")

    def _assert_square(self, mod: nn.Module, attrname: str) -> int:
        attrval: Tuple[int, ...] = getattr(mod, attrname)
        if isinstance(attrval, int):
            # MaxPool2d
            return attrval
        assert (attrval[0] == attrval[1]) and len(
            attrval
        ) == 2, f"{attrname} for {self.name} must be square, found {attrval}"
        return attrval[0]

    def module_info(self) -> str:
        if self.kernel_size is None:
            return self.module_class
        else:
            return (
                f"{self.module_class} [{self.kernel_size},{self.stride},{self.padding}]"
            )

    def has_children(self) -> bool:
        return len(list(self.module.children())) > 0

    def conv2d_complexity(self) -> Optional[int]:
        assert self.output_shape is not None
        return (
            (
                self.input_shape.tensor_shape[1]  # num_input_filters
                * self.output_shape.tensor_shape[1]  # num_output_filters
                * self.output_shape.tensor_shape[-1]  # output_height
                * self.output_shape.tensor_shape[-2]  # output_width
            )
            if self.kernel_size is not None
            else None
        )


@attr.s(auto_attribs=True, eq=False)
class ConvInfo:
    jump: int
    start: float
    receptive_field: int
    conv_stage: bool = True
    grad_receptive_field: Tuple[int, ...] = (-1,)

    @classmethod
    def for_module(cls, mod: ModuleInfo, prev: ModuleInfo):
        prev_conv_info = prev.conv_info
        assert (
            prev_conv_info is not None
        ), f"{prev} has no conv_info, while computing {mod}"

        if mod_isinstance(mod.module, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d)):
            assert (
                mod.kernel_size is not None
                and mod.stride is not None
                and mod.padding is not None
            )

            # https://medium.com/mlreview/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807
            return ConvInfo(
                jump=prev_conv_info.jump * mod.stride,
                receptive_field=(
                    prev_conv_info.receptive_field
                    + (mod.kernel_size - 1) * prev_conv_info.jump
                ),
                start=(
                    prev_conv_info.start
                    + ((mod.kernel_size - 1) / 2 - mod.padding) * prev_conv_info.jump
                ),
            )
        elif mod_isinstance(mod.module, nn.ConvTranspose2d):
            return ConvInfo(jump=0, receptive_field=0, start=0, conv_stage=False)
        else:
            return ConvInfo(
                jump=prev_conv_info.jump,
                start=prev_conv_info.start,
                receptive_field=prev_conv_info.receptive_field,
                conv_stage=False,
            )


@attr.s(auto_attribs=True)
class ModuleInfoIndex:
    by_name: Dict[str, ModuleInfo] = attr.ib(factory=dict)
    by_input_order: Dict[int, ModuleInfo] = attr.ib(factory=dict)
    by_output_order: Dict[int, ModuleInfo] = attr.ib(factory=dict)

    def update(self, module_info: ModuleInfo):
        self.by_name[module_info.name] = module_info
        self.by_input_order[module_info.input_order] = module_info
        if module_info.output_order is not None:
            self.by_output_order[module_info.output_order] = module_info

    def __len__(self):
        return len(self.by_name)


def get_pre_hook(module, name: str, infos: ModuleInfoIndex) -> Callable:

    """
    Instantiates ModuleInfo() for the module.

    Executes before forward()
    """

    def hook(module: nn.Module, input):
        if name in infos.by_name:
            return

        msum = ModuleInfo(
            name=name,
            module=module,
            input_order=len(infos),
            # Input is a tuple of size 1, always?
            input_shape=shape(*input),
        )
        if msum.input_order > 0:
            msum.conv_info = ConvInfo.for_module(
                msum, prev=infos.by_input_order[msum.input_order - 1]
            )
        infos.update(msum)

    return hook


def get_fwd_hook(
    name: str,
    infos: ModuleInfoIndex,
    inputs: List[Tensor],
    output_order: List[str],
    grad_receptive_field: bool,
    pbar: Optional[tqdm] = None,
) -> Callable:
    """
    Adds output shape to the ModuleInfo() previously set up by the pre hook.

    Runs after forward()
    """

    # We need this weird ~output_order~ list so that hooks can mutate some
    # common variable and compute their order of invocation. We cannot
    # use a variable local to get_hook() because it would get reset each
    # time -- it needs to belong to register_hooks()'s scope
    def hook(module: nn.Module, _input, output):
        msum = infos.by_name[name]

        if msum.output_shape is not None:
            # already handled
            return

        msum.output_shape = shape(output)
        msum.output_order = len(output_order)
        output_order.append(name)
        infos.update(msum)

        if grad_receptive_field:
            if msum.conv_info.conv_stage:
                # Note: we're only computing the receptive field wrt the first input
                msum.conv_info.grad_receptive_field = compute_grad_receptive_field(
                    msum, input=inputs[0], output=output, infos=infos
                )
            else:
                prev = infos.by_output_order[msum.output_order - 1]
                msum.conv_info.grad_receptive_field = (
                    prev.conv_info.grad_receptive_field
                )

        if pbar:
            pbar.update()

    return hook


def init_for_grad_receptive_field(m: nn.Module):
    assert m.weight is not None
    nn.init.constant_(cast(Tensor, m.weight), val=1.0)
    if m.bias is not None:
        nn.init.constant_(cast(Tensor, m.bias), val=0.0)


def prepare_model_for_grad_receptive_field(model: nn.Module) -> nn.Module:
    # See:
    # https://github.com/rogertrullo/Receptive-Field-in-Pytorch/blob/master/Receptive_Field.ipynb

    # make copy
    model = deepcopy(model)
    model.apply(modify_for_grad_receptive_field)
    return model


def modify_for_grad_receptive_field(module: nn.Module):
    for name, m in module.named_children():
        if mod_isinstance(m, nn.Conv2d):
            init_for_grad_receptive_field(m)
        elif mod_isinstance(m, nn.MaxPool2d):
            # change maxpool to avgpool
            replacement = nn.AvgPool2d(
                kernel_size=cast(int, m.kernel_size),
                stride=cast(int, m.stride),
                padding=cast(int, m.padding),
                ceil_mode=cast(bool, m.ceil_mode),
            )
            module.add_module(name, replacement)
        elif mod_isinstance(m, nn.Dropout):
            # turn off Dropout
            # https://discuss.pytorch.org/t/how-to-freeze-bn-layers-while-training-the-rest-of-network-mean-and-var-wont-freeze/89736/10
            m.eval()
        elif mod_isinstance(m, nn.BatchNorm2d):
            # turn off batchnorm
            # https://discuss.pytorch.org/t/how-to-close-batchnorm-when-using-torchvision-models/21812/2
            if hasattr(m, "reset_parameters"):
                cast(Callable, m.reset_parameters)()
            m.eval()
            init_for_grad_receptive_field(m)


def compute_grad_receptive_field(
    mod: ModuleInfo, input: Tensor, output: Tensor, infos: ModuleInfoIndex
) -> Tuple[int, ...]:

    if not isinstance(output, Tensor):
        # We cannot backward() from this. Skip, we just won't be able
        # to compute the receptive field here without some other
        # affordance to combine the tensors in this output into a
        # "loss"
        return (-1,)

    # https://github.com/rogertrullo/Receptive-Field-in-Pytorch/blob/master/Receptive_Field.ipynb
    fake_grad = torch.zeros(output.shape)
    # batch=0, channel=0
    center_pos = (0, 0, *[i // 2 for i in output.shape[2:]])
    fake_grad[center_pos] = 1

    # zero_grad everything before and including this module, since backward() accumulates
    for i in range(0, mod.input_order + 1):
        infos.by_input_order[i].module.zero_grad()

    # retain_graph so we can run this multiple times and not drop the
    # intermediate results from forward()
    output.backward(gradient=fake_grad, retain_graph=True)

    # Find the extent of pixels affected; drop batch/channel
    nonzero_idxs = input.grad.nonzero(as_tuple=True)[2:]
    rf_dims = [(d.max() - d.min() + 1).item() for d in nonzero_idxs]
    return tuple(rf_dims)


@contextmanager
def register_hooks(
    model,
    infos: ModuleInfoIndex,
    bs: int,
    inputs: List[Tensor],
    grad_receptive_field: bool,
    use_pbar: bool,
):
    hooks = []
    output_order: list = []
    progbar = tqdm() if use_pbar else None

    try:
        # apply() doesn't give us names
        for idx, (module_name, module) in enumerate(model.named_modules()):

            pre_hk = get_pre_hook(
                module,
                name=module_name,
                infos=infos,
            )
            hk = get_fwd_hook(
                name=module_name,
                infos=infos,
                output_order=output_order,
                inputs=inputs,
                grad_receptive_field=grad_receptive_field,
                pbar=progbar,
            )
            hooks.append(module.register_forward_pre_hook(pre_hk))
            hooks.append(module.register_forward_hook(hk))

        if progbar is not None:
            progbar.reset(len(hooks) // 2)

        yield
    finally:
        for h in hooks:
            h.remove()

        if progbar is not None:
            progbar.close()


def summary(
    model: nn.Module,
    *inputs: Any,
    batch_size: int = -1,
    extract_input_tensor: Optional[Callable[[Any], Tensor]] = None,
    include_input_shape: bool = True,
    include_conv_info: bool = True,
    grad_receptive_field: bool = True,
    bytes_per_float: int = 4,
    use_pbar: bool = True,
):

    assert len(inputs) > 0, "Has inputs"

    input_tensors = (
        [extract_input_tensor(i) for i in inputs]
        if extract_input_tensor is not None
        else [cast(Tensor, i) for i in inputs]
    )
    input_sizes: List[List[int]] = [list(t.shape) for t in input_tensors]

    assert all(
        sz[0] >= 2 for sz in input_sizes
    ), "_Each_ input must have batch_size >= 2 (for batchnorm)"

    module_infos = ModuleInfoIndex()

    if grad_receptive_field:
        # If we're numerically computing the Receptive Field, we will
        # be messing with the model
        model = prepare_model_for_grad_receptive_field(model)
        for i in input_tensors:
            i.requires_grad = True

    with register_hooks(
        model,
        infos=module_infos,
        bs=batch_size,
        inputs=input_tensors,
        grad_receptive_field=grad_receptive_field,
        use_pbar=use_pbar,
    ):
        model(*inputs)

    print(
        make_output(
            infos=module_infos,
            batch_size=batch_size,
            input_sizes=input_sizes,
            include_input_shape=include_input_shape,
            include_conv_info=include_conv_info,
            grad_receptive_field=grad_receptive_field,
            bytes_per_float=bytes_per_float,
        )
    )


def make_output(
    infos: ModuleInfoIndex,
    batch_size: int,
    input_sizes: Iterable[Union[torch.Size, Iterable[int]]],
    include_input_shape: bool,
    include_conv_info: bool,
    grad_receptive_field: bool,
    bytes_per_float: int,
) -> str:

    layer_table_cols = make_layer_table(
        infos,
        include_input_shape=include_input_shape,
        include_conv_info=include_conv_info,
        grad_receptive_field=grad_receptive_field,
    )

    summary_table = make_summary_table(
        infos,
        bytes_per_float=bytes_per_float,
        batch_size=batch_size,
        input_sizes=input_sizes,
    )

    note = (
        [
            "",
            "** Note: grad_receptive_field=True, Model modified: Conv2d init, MaxPool2d->AvgPool2d, Dropout/Batchnorm off",
        ]
        if grad_receptive_field
        else []
    )

    return "\n".join(["", layer_table_cols, "", summary_table] + note)


def nullsafe_format_number(n) -> Optional[str]:
    return format_number(n) if n is not None else None


ColFunc = Callable[[ModuleInfo], Any]


def make_layer_table(
    infos: ModuleInfoIndex,
    include_input_shape: bool,
    include_conv_info: bool,
    grad_receptive_field: bool,
) -> str:
    layer_table_cols: Dict[str, Optional[ColFunc]] = {
        "Input Shape": (
            (lambda m: pformat(m.input_shape, width=SHAPE_COL_WIDTH))
            if include_input_shape
            else None
        ),
        "Output Shape": lambda m: pformat(m.output_shape, width=SHAPE_COL_WIDTH),
        "Input size": lambda m: nullsafe_format_number(m.input_shape.size),
        "Num params": (
            lambda m: nullsafe_format_number(m.num_params) if m.num_params > 0 else None
        ),
        "Conv2d complexity": lambda m: nullsafe_format_number(m.conv2d_complexity()),
        "Start": (
            (
                lambda m: nullsafe_format_number(m.conv_info.start)
                if m.conv_info is not None and m.conv_info.conv_stage
                else None
            )
            if include_conv_info
            else None
        ),
        "Jump": (
            (
                lambda m: m.conv_info.jump
                if m.conv_info is not None and m.conv_info.conv_stage
                else None
            )
            if include_conv_info
            else None
        ),
        "ReceptiveField": (
            (
                lambda m: m.conv_info.receptive_field
                if m.conv_info is not None
                and (m.conv_info.conv_stage or m.has_children())
                else None
            )
            if include_conv_info
            else None
        ),
        "GradReceptiveField": (
            (
                lambda m: m.conv_info.grad_receptive_field
                if m.conv_info is not None
                and (m.conv_info.conv_stage or m.has_children())
                else None
            )
            if grad_receptive_field
            else None
        ),
    }

    labeled_summaries = create_tree_labels(
        sorted(infos.by_name.values(), key=lambda m: m.input_order)
    )
    rows = []
    for m, label in labeled_summaries:
        r = [label]
        for _, fun in layer_table_cols.items():
            if fun is not None:
                colval = fun(m)
                r.append(colval if colval is not None else "")
        rows.append(r)

    return tabulate.tabulate(
        rows,
        headers=["Layer (type,kernel,stride,padding)"]
        + [k for k, v in layer_table_cols.items() if v is not None],
        floatfmt=",.0f",
        tablefmt="psql",
        disable_numparse=True,
    )


def make_summary_table(
    infos: ModuleInfoIndex,
    input_sizes: Iterable[Union[torch.Size, Iterable[int]]],
    batch_size: int,
    bytes_per_float: int,
) -> str:
    total_params = 0
    trainable_params = 0

    # We use a dict to uniquify tensors among the intermediate
    # (ie. all inputs + last output) feature maps : the key is
    # id(tensor)
    intermediates: Dict[int, int] = {}
    for m in infos.by_name.values():
        total_params += m.num_params
        if m.trainable:
            trainable_params += m.num_params

        # We prefer to count inputs because there may be unused
        # outputs.
        intermediates.update(m.input_shape.tensor_sizes)

    # Capture only the last output separately
    last_module = max(infos.by_name.values(), key=lambda m: m.output_order or -1)
    assert last_module.output_shape is not None
    intermediates.update(last_module.output_shape.tensor_sizes)

    total_input_size = abs(prod(sum(input_sizes, [])) * batch_size * bytes_per_float)
    total_intermediate = sum(intermediates.values())
    total_intermediate_size = abs(
        2.0 * total_intermediate * bytes_per_float
    )  # x2 for gradients
    total_params_size = abs(total_params * bytes_per_float)
    total_size = total_params_size + total_intermediate_size + total_input_size

    summary_table = [
        ["Total params", nullsafe_format_number(total_params)],
        ["Trainable params", nullsafe_format_number(trainable_params)],
        [
            "Non-trainable params",
            nullsafe_format_number(total_params - trainable_params),
        ],
        ["Input size", format_size(total_input_size, binary=True)],
        [
            "Intermediates size",
            format_size(total_intermediate_size, binary=True),
        ],
        ["Params size", format_size(total_params_size, binary=True)],
        [
            "Estimated Total Size",
            format_size(total_size, binary=True),
        ],
    ]

    return tabulate.tabulate(
        summary_table, floatfmt=",.1f", tablefmt="psql", disable_numparse=True
    )


def parse_tree(
    infos: Iterable[ModuleInfo],
) -> Dict[ModuleInfo, Tuple[List[str], Optional[str], str]]:
    # not all nodes in the tree have infos. this function computes
    # the ancestors for each module that do have infos, and the
    # "leaf name" which is the path from the nearest ancestor.

    nodes = set(s.name for s in infos)

    tree: Dict[ModuleInfo, Tuple[List[str], Optional[str], str]] = {}
    for s in infos:
        if s.name == "":
            tree[s] = ([], None, "")
        else:
            parts = s.name.split(".")
            ancestors = []
            last_ancestor_i = 0
            for i in range(len(parts)):
                path = ".".join(parts[:i])
                if path in nodes:
                    ancestors.append(path)
                    last_ancestor_i = i
            leaf = ".".join(parts[last_ancestor_i:])
            tree[s] = (ancestors, ancestors[-1], leaf)
    return tree


def create_tree_labels(
    infos: Iterable[ModuleInfo],
) -> Generator[Tuple[ModuleInfo, str], None, None]:

    vert_stem, branch, end_branch = (
        "\u2502 ",  # │
        "\u251c\u2500\u2500 ",  # ├──
        "\u2514\u2500\u2500 ",  # └──
    )

    tree = parse_tree(infos)

    num_children: Dict[Optional[str], int] = defaultdict(lambda: 0)
    for s in infos:
        parent = tree[s][1]
        num_children[parent] += 1

    for s in infos:
        ancestors, parent, leaf = tree[s]
        if not len(ancestors):
            yield (s, s.module_class)
        else:
            child_counts = [num_children[a] for a in ancestors]
            stems = "".join(vert_stem if c > 0 else " " for c in child_counts[:-1])
            br = end_branch if child_counts[-1] == 1 else branch
            yield (s, f"{stems}{br}{leaf} ({s.module_info()})")

        num_children[parent] -= 1
