from __future__ import annotations

from collections import defaultdict
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
    Union,
)

import attr
import tabulate
import torch
from humanfriendly import format_number, format_size  # type: ignore
from shapely import Shape, shape

SHAPE_COL_WIDTH = 50
tabulate.PRESERVE_WHITESPACE = True


def prod(x: Iterable) -> float:
    return reduce(mul, x, 1.0)


def get_tensor_size(input: torch.Tensor, idx: int) -> torch.Size:
    return input.size()


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
    module: torch.nn.Module
    input_shape: Shape

    # input_order and output_order will be different in the case of
    # nested modules: the "wrapper" module will appear first in input
    # order, but last in output order. Similarly the innermost module
    # will appear first in output_order.
    input_order: int

    output_shape: Optional[Shape] = None
    output_order: int = -1

    module_class: str = attr.ib(init=False)
    num_params: int = attr.ib(init=False, default=0)
    trainable: bool = attr.ib(init=False, default=False)

    kernel_size: Optional[int] = attr.ib(init=False, default=None)
    padding: Optional[int] = attr.ib(init=False, default=None)
    stride: Optional[int] = attr.ib(init=False, default=None)

    _conv_info: Optional[ConvInfo] = attr.ib(init=False, default=None)

    def __attrs_post_init__(self):
        self.module_class = self.module.__class__.__name__

        if hasattr(self.module, "weight") and hasattr(self.module.weight, "size"):
            self.num_params += prod(self.module.weight.size())
            self.trainable = self.module.weight.requires_grad

        if hasattr(self.module, "bias") and hasattr(self.module.bias, "size"):
            self.num_params += prod(self.module.bias.size())

        if isinstance(self.module, (torch.nn.Conv2d, torch.nn.MaxPool2d)):
            self.kernel_size = self._assert_square(self.module, "kernel_size")
            self.stride = self._assert_square(self.module, "stride")
            self.padding = self._assert_square(self.module, "padding")

    def _assert_square(self, mod: torch.nn.Module, attrname: str) -> int:
        attrval: Tuple[int, ...] = getattr(mod, attrname)
        if isinstance(attrval, int):
            # MaxPool2d
            return attrval
        assert (attrval[0] == attrval[1]) and len(
            attrval
        ) == 2, f"{attrname} for {self.name} must be square, found {attrval}"
        return attrval[0]

    @property
    def conv_info(self) -> Optional[ConvInfo]:
        if self._conv_info is not None:
            return self._conv_info
        elif self.input_order == 0:
            return ConvInfo(start=0.5, jump=1, receptive_field=1)
        else:
            return None

    @conv_info.setter
    def conv_info(self, val: ConvInfo):
        self._conv_info = val

    def module_info(self) -> str:
        if self.kernel_size is None:
            return self.module_class
        else:
            return (
                f"{self.module_class} [{self.kernel_size},{self.stride},{self.padding}]"
            )


@attr.s(auto_attribs=True, eq=False)
class ConvInfo:
    jump: int
    start: float
    receptive_field: int
    conv_stage: bool = True

    @classmethod
    def for_module(cls, mod: ModuleInfo, prev: ModuleInfo):
        prev_conv_info = prev.conv_info
        assert (
            prev_conv_info is not None
        ), f"{prev} has no conv_info, while computing {mod}"

        if isinstance(mod.module, (torch.nn.Conv2d, torch.nn.MaxPool2d)):
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
        elif isinstance(mod.module, torch.nn.ConvTranspose2d):
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

    def add(self, module_info: ModuleInfo):
        self.by_name[module_info.name] = module_info
        self.by_input_order[module_info.input_order] = module_info

    def __len__(self):
        return len(self.by_name)


def get_pre_hook(
    module, name: str, infos: ModuleInfoIndex, include_conv_info: bool
) -> Callable:

    """
    Instantiates ModuleInfo() for the module.

    Executes before forward()
    """

    def hook(module, input):
        if name in infos.by_name:
            return

        msum = ModuleInfo(
            name=name,
            module=module,
            input_order=len(infos),
            # Input is a tuple of size 1, always?
            input_shape=shape(*input),
        )

        infos.add(msum)
        if include_conv_info and msum.input_order > 0:
            msum.conv_info = ConvInfo.for_module(
                msum, prev=infos.by_input_order[msum.input_order - 1]
            )

    return hook


def get_hook(name: str, infos: ModuleInfoIndex, output_order: list) -> Callable:
    """
    Adds output shape to the ModuleInfo() previously set up by the pre hook.

    Runs after forward()
    """

    # We need this weird ~output_order~ list so that hooks can mutate some
    # common variable and compute their order of invocation. We cannot
    # use a variable local to get_hook() because it would get reset each
    # time -- it needs to belong to register_hooks()'s scope
    def hook(module, input, output):
        msum = infos.by_name[name]

        if msum.output_shape is not None:
            # already handled
            return

        msum.output_shape = shape(output)
        msum.output_order = len(output_order)

        output_order.append(name)

    return hook


def register_hooks(
    model, infos: ModuleInfoIndex, bs: int, include_conv_info: bool
) -> List:
    hooks = []
    output_order: list = []

    # apply() doesn't give us names
    for idx, (module_name, module) in enumerate(model.named_modules()):

        pre_hk = get_pre_hook(
            module,
            name=module_name,
            infos=infos,
            include_conv_info=include_conv_info,
        )
        hk = get_hook(
            name=module_name,
            infos=infos,
            output_order=output_order,
        )
        hooks.append(module.register_forward_pre_hook(pre_hk))
        hooks.append(module.register_forward_hook(hk))

    return hooks


def summary(
    model: torch.nn.Module,
    *inputs: Any,
    batch_size: int = -1,
    get_input_size: Callable[[Any, int], torch.Size] = get_tensor_size,
    include_input_shape: bool = False,
    include_conv_info: bool = False,
    bytes_per_float: int = 4,
    eval: bool = True,
) -> Tuple[int, int]:

    assert len(inputs) > 0, "Has inputs"

    input_sizes: List[List[int]] = [
        list(get_input_size(inp, i)) for i, inp in enumerate(inputs)
    ]
    assert len(input_sizes) == len(
        inputs
    ), f"Input sizes from get_input_sizes must match the number of inputs: {len(inputs)=} {len(input_sizes)=}"
    assert all(
        sz[0] >= 2 for sz in input_sizes
    ), "_Each_ input must have batch_size >= 2 (for batchnorm)"

    module_infos = ModuleInfoIndex()
    restore_training = model.training
    try:
        hooks = register_hooks(
            model,
            infos=module_infos,
            bs=batch_size,
            include_conv_info=include_conv_info,
        )
        if eval:
            model.eval()
        with torch.no_grad():
            model(*inputs)
    finally:
        model.train(restore_training)
        for h in hooks:
            h.remove()

    summary_table, total_params, trainable_params = make_output(
        infos=module_infos,
        batch_size=batch_size,
        input_sizes=input_sizes,
        include_input_shape=include_input_shape,
        include_conv_info=include_conv_info,
        bytes_per_float=bytes_per_float,
    )
    print(summary_table)
    return total_params, trainable_params


def make_output(
    infos: ModuleInfoIndex,
    batch_size: int,
    input_sizes: Iterable[Union[torch.Size, Iterable[int]]],
    include_input_shape: bool,
    include_conv_info: bool,
    bytes_per_float: int,
) -> Tuple[str, int, int]:

    total_params = 0
    trainable_params = 0
    layer_table = []
    layer_table_header = [
        h
        for h in [
            "Layer (type,kernel,stride,padding)",
            "Input Shape" if include_input_shape else None,
            "Output Shape",
            "Input size",
            "Num params",
            "Conv2d complexity",
            *(["Start", "Jump", "ReceptiveField"] if include_conv_info else []),
        ]
        if h is not None
    ]

    # We use a dict to uniquify tensors among the intermediate
    # (ie. all inputs + last output) feature maps : the key is
    # id(tensor)
    intermediates: Dict[int, int] = {}
    labeled_summaries = create_tree_labels(
        sorted(infos.by_name.values(), key=lambda m: m.input_order)
    )
    for m, label in labeled_summaries:
        assert m.output_shape is not None

        conv2d_complexity = (
            (
                m.input_shape.tensor_shape[1]  # num_input_filters
                * m.output_shape.tensor_shape[1]  # num_output_filters
                * m.output_shape.tensor_shape[-1]  # output_height
                * m.output_shape.tensor_shape[-2]  # output_width
            )
            if m.kernel_size is not None
            else None
        )
        row = [
            c
            for c in [
                label,
                (
                    pformat(m.input_shape, width=SHAPE_COL_WIDTH)
                    if include_input_shape
                    else None
                ),
                pformat(m.output_shape, width=SHAPE_COL_WIDTH),
                format_number(m.input_shape.size),
                format_number(m.num_params),
                (
                    format_number(conv2d_complexity)
                    if conv2d_complexity is not None
                    else ""
                ),
                *(
                    [
                        format_number(m.conv_info.start),
                        m.conv_info.jump,
                        m.conv_info.receptive_field,
                    ]
                    if include_conv_info
                    and m.conv_info is not None
                    and m.conv_info.conv_stage
                    else []
                ),
            ]
            if c is not None
        ]
        layer_table.append(row)
        total_params += m.num_params
        if m.trainable:
            trainable_params += m.num_params

        # We prefer to count inputs because there may be unused
        # outputs.
        intermediates.update(m.input_shape.tensor_sizes)

    # Capture only the last output separately
    last_module = max(infos.by_name.values(), key=lambda m: m.output_order)
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
        ["Total params", format_number(total_params)],
        ["Trainable params", format_number(trainable_params)],
        ["Non-trainable params", format_number(total_params - trainable_params)],
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

    summary_str = "\n".join(
        [
            tabulate.tabulate(
                layer_table,
                headers=layer_table_header,
                floatfmt=",.0f",
                tablefmt="psql",
                disable_numparse=True,
            ),
            "",
            tabulate.tabulate(
                summary_table, floatfmt=",.1f", tablefmt="psql", disable_numparse=True
            ),
        ]
    )

    return summary_str, total_params, trainable_params


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
