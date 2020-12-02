from collections import defaultdict
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
import numpy as np
import tabulate
import torch
from humanfriendly import format_number, format_size
from shapely import shape, Shape

gpu_if_available = "cuda:0" if torch.cuda.is_available() else "cpu"
SHAPE_COL_WIDTH = 50


tabulate.PRESERVE_WHITESPACE = True


def get_tensor_size(input: torch.Tensor, idx: int) -> torch.Size:
    return input.size()


def make_random_input(
    *shape: int, dtype=torch.FloatTensor, device=gpu_if_available, bs: Optional[int] = 2
):
    """
    Chooses bs=2 by default for batchnorm. bs=None means to squeeze out that dimensino
    """
    assert len(shape) == 3, "Expected shape to be (C, H, W), got {shape=}"
    shape = (bs, *shape) if bs is not None else shape
    return torch.rand(*shape).type(dtype).to(device=device)


@attr.s(eq=False)
class ModuleSummary:
    name: str = attr.ib()
    module_class: str = attr.ib()
    input_order: int = attr.ib()
    input_shape: Shape = attr.ib()
    output_order: int = attr.ib(default=-1)
    output_shape: Optional[Shape] = attr.ib(default=None)
    trainable: bool = attr.ib(default=False)
    num_params: int = attr.ib(default=0)
    kernel_size: Optional[Tuple[int, int]] = attr.ib(default=None)


def get_pre_hook(module, name: str, summaries: Dict[str, ModuleSummary]) -> Callable:
    def hook(module, input):
        if name in summaries:
            return

        msum = ModuleSummary(
            name=name,
            module_class=module.__class__.__name__,
            input_order=len(summaries),
            # Input is a tuple of size 1, always?
            input_shape=shape(*input),
        )

        if hasattr(module, "weight") and hasattr(module.weight, "size"):
            msum.num_params += np.prod(module.weight.size())
            msum.trainable = module.weight.requires_grad
        if hasattr(module, "bias") and hasattr(module.bias, "size"):
            msum.num_params += np.prod(module.bias.size())

        if isinstance(module, torch.nn.Conv2d):
            msum.kernel_size = module.kernel_size

        summaries[name] = msum

    return hook


def get_hook(
    name: str, summaries: Dict[str, ModuleSummary], output_order: list
) -> Callable:
    """We need this weird output_order list so that hooks can mutate some
    common variable and compute their order of invocation. We cannot
    use a variable local to get_hook() because it would get reset each
    time -- it needs to belong to register_hooks()'s scope

    """

    def hook(module, input, output):
        msum = summaries[name]
        if msum.output_shape is not None:
            return

        msum.output_shape = shape(output)
        msum.output_order = len(output_order)
        output_order.append(name)

    return hook


def register_hooks(model, summaries: Dict[str, ModuleSummary], bs: int) -> List:
    hooks = []
    output_order: list = []
    for idx, (module_name, module) in enumerate(model.named_modules()):
        if isinstance(module, (torch.nn.Sequential, torch.nn.ModuleList)):
            continue
        pre_hk = get_pre_hook(module, name=module_name, summaries=summaries)
        hk = get_hook(name=module_name, summaries=summaries, output_order=output_order)
        hooks.append(module.register_forward_pre_hook(pre_hk))
        hooks.append(module.register_forward_hook(hk))
    return hooks


def summary(
    model: torch.nn.Module,
    *inputs: Any,
    batch_size: int = -1,
    get_input_size: Callable[[Any, int], torch.Size] = get_tensor_size,
    include_input_shape: bool = False,
    bytes_per_float: int = 4,
    eval: bool = True,
) -> Tuple[int, int]:

    input_sizes: List[List[int]] = [
        list(get_input_size(inp, i)) for i, inp in enumerate(inputs)
    ]
    assert len(input_sizes) == len(
        inputs
    ), f"Input sizes from get_input_sizes must match the number of inputs: {len(inputs)=} {len(input_sizes)=}"
    assert all(
        sz[0] >= 2 for sz in input_sizes
    ), "_Each_ input must have batch_size >= 2 (for batchnorm)"

    module_summaries: Dict[str, ModuleSummary] = {}
    hooks: List = []

    restore_training = model.training
    try:
        register_hooks(model, summaries=module_summaries, bs=batch_size)
        if eval:
            model.eval()
        with torch.no_grad():
            model(*inputs)
    finally:
        model.train(restore_training)
        for h in hooks:
            h.remove()

    summary_table, total_params, trainable_params = make_output(
        summaries=module_summaries,
        batch_size=batch_size,
        input_sizes=input_sizes,
        include_input_shape=include_input_shape,
        bytes_per_float=bytes_per_float,
    )
    print(summary_table)
    return total_params, trainable_params


def make_output(
    summaries: Dict[str, ModuleSummary],
    batch_size: int,
    input_sizes: List[torch.Size],
    include_input_shape: bool,
    bytes_per_float: int,
) -> Tuple[str, int, int]:

    total_params = 0
    trainable_params = 0
    layer_table = []
    layer_table_header = [
        h
        for h in [
            "Layer (type)",
            "Input Shape" if include_input_shape else None,
            "Output Shape",
            "Input size",
            "Num params",
            "Conv2d complexity (C_i * C_o * H_o * W_o))",
        ]
        if h is not None
    ]

    # We use a dict to uniquify tensors among the intermediate
    # (ie. all inputs + last output) feature maps : the key is
    # id(tensor)
    intermediates: Dict[int, int] = {}
    labeled_summaries = create_tree_labels(
        sorted(summaries.values(), key=lambda m: m.input_order)
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
                    pformat(m.input_shape.dump(), width=SHAPE_COL_WIDTH)
                    if include_input_shape
                    else None
                ),
                pformat(m.output_shape.dump(), width=SHAPE_COL_WIDTH),
                format_number(m.input_shape.size),
                format_number(m.num_params),
                format_number(conv2d_complexity)
                if conv2d_complexity is not None
                else None,
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
    last_module = max(summaries.values(), key=lambda m: m.output_order)
    assert last_module.output_shape is not None
    intermediates.update(last_module.output_shape.tensor_sizes)

    total_input_size = abs(np.prod(sum(input_sizes, [])) * batch_size * bytes_per_float)
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
        ["Intermediates size", format_size(total_intermediate_size, binary=True),],
        ["Params size", format_size(total_params_size, binary=True)],
        ["Estimated Total Size", format_size(total_size, binary=True),],
    ]

    summary_str = "\n".join(
        [
            tabulate.tabulate(
                layer_table,
                headers=layer_table_header,
                floatfmt=",.0f",
                tablefmt="psql",
            ),
            "",
            tabulate.tabulate(summary_table, floatfmt=",.1f", tablefmt="psql"),
        ]
    )

    return summary_str, total_params, trainable_params


def parse_tree(
    summaries: Iterable[ModuleSummary],
) -> Dict[ModuleSummary, Tuple[List[str], Optional[str], str]]:
    # not all nodes in the tree have summaries. this function computes
    # the ancestors for each module that do have summaries, and the
    # "leaf name" which is the path from the nearest ancestor.

    nodes = set(s.name for s in summaries)

    tree: Dict[ModuleSummary, Tuple[List[str], Optional[str], str]] = {}
    for s in summaries:
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
    summaries: Iterable[ModuleSummary],
) -> Generator[Tuple[ModuleSummary, str], None, None]:

    vert_stem, branch, end_branch, space = (
        "\u2502 ",  # │
        "\u251c\u2500\u2500 ",  # ├──
        "\u2514\u2500\u2500 ",  # └──
        " ",
    )

    tree = parse_tree(summaries)

    num_children: Dict[Optional[str], int] = defaultdict(lambda: 0)
    for s in summaries:
        parent = tree[s][1]
        num_children[parent] += 1

    for s in summaries:
        ancestors, parent, leaf = tree[s]
        if not len(ancestors):
            yield (s, s.module_class)
        else:
            child_counts = [num_children[a] for a in ancestors]
            stems = "".join(vert_stem if c > 0 else " " for c in child_counts[:-1])
            br = end_branch if child_counts[-1] == 1 else branch
            class_info = (
                s.module_class
                if s.kernel_size is None
                else f"{s.module_class} [{s.kernel_size}]"
            )
            yield (s, f"{stems}{br}{leaf} ({class_info})")

        num_children[parent] -= 1
