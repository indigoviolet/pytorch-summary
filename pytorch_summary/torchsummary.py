import torch
import numpy as np

from tabulate import tabulate
import attr

from typing import List, Dict, Callable, Any, Iterable, Optional, Union, Tuple
from pprint import pformat


gpu_if_available = "cuda:0" if torch.cuda.is_available() else "cpu"
SHAPE_COL_WIDTH = 50


def get_tensor_size(input: torch.Tensor, idx: int) -> torch.Size:
    return input.size()


def make_random_input(
    *shape: int, dtype=torch.FloatTensor, device=gpu_if_available, bs: int = 2
):
    return torch.rand(bs, *shape).type(dtype).to(device=device)


@attr.s(auto_attribs=True)
class Shape:
    value: Any
    maxlen: int = 3
    size: Optional[int] = None

    def __attrs_post_init__(self):
        if hasattr(self.value, "shape"):
            self.size = abs(np.prod(self._get_torch_size_as_list(self.value.shape)))

    def _get_torch_size_as_list(self, arg) -> list:
        return [-1] + list(arg[1:]) if len(arg) > 3 else list(arg)

    def dump(self):
        return self._dump(self.value, maxlen=self.maxlen)

    def _dump(self, arg, maxlen) -> Any:
        t = type(arg)
        if t is dict:
            return {k: self._dump(v, maxlen) for k, v in arg.items()}
        elif t is tuple or (t is list and maxlen >= len(arg)):
            return t(self._dump(v, maxlen) for v in arg)
        elif t is list and maxlen <= len(arg):
            return (f"L({len(arg)})", [self._dump(v, maxlen) for v in arg[:maxlen]])
        elif hasattr(arg, "shape"):
            return self._dump(arg.shape, maxlen)
        elif t is torch.Size:
            return self._get_torch_size_as_list(arg)
        elif arg.__class__.__module__ == "builtins":
            return arg
        else:
            return classname(arg)


def classname(arg) -> str:
    return f"{arg.__class__.__module__}.{arg.__class__.__qualname__}"


def summary(
    model: torch.nn.Module,
    *inputs: Any,
    batch_size: int = -1,
    get_input_size: Callable[[Any, int], torch.Size] = get_tensor_size,
    include_input_shape: bool = False,
):
    result = summary_string(
        model,
        *inputs,
        batch_size=batch_size,
        get_input_size=get_input_size,
        include_input_shape=include_input_shape,
    )
    print(result)


def shape(arg, maxlen=3) -> Shape:
    return Shape(arg, maxlen=maxlen)


@attr.s
class ModuleSummary:
    name: str = attr.ib()
    sort_key: int = attr.ib()
    input_shape: Shape = attr.ib()
    output_shape: Shape = attr.ib()
    trainable: bool = attr.ib(init=False, default=False)
    num_params: int = attr.ib(init=False, default=0)


def get_hook(
    module, idx: int, name: str, summaries: Dict[str, ModuleSummary], bs: int
) -> Callable:
    def hook(module, input, output):
        # Input is a tuple of size 1, always?
        msum = ModuleSummary(
            name=f"{name or '?'} ({module.__class__.__name__})",
            sort_key=idx,
            input_shape=shape(*input),
            output_shape=shape(output),
        )

        if hasattr(module, "weight") and hasattr(module.weight, "size"):
            msum.num_params += np.prod(module.weight.size())
            msum.trainable = module.weight.requires_grad
        if hasattr(module, "bias") and hasattr(module.bias, "size"):
            msum.num_params += np.prod(module.bias.size())

        summaries[msum.name] = msum

    return hook


def register_hooks(model, summaries: Dict[str, ModuleSummary], bs: int) -> List:
    hooks = []
    for idx, (module_name, module) in enumerate(model.named_modules()):
        if isinstance(module, (torch.nn.Sequential, torch.nn.ModuleList)):
            continue
        hk = get_hook(module, name=module_name, idx=idx, summaries=summaries, bs=bs)
        hooks.append(module.register_forward_hook(hk))
    return hooks


def summary_string(
    model: torch.nn.Module,
    *inputs: Any,
    batch_size: int = -1,
    get_input_size: Callable[[Any, int], torch.Size] = get_tensor_size,
    include_input_shape: bool = False,
) -> str:

    input_sizes = [get_input_size(inp, i) for i, inp in enumerate(inputs)]
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
        # make a forward pass
        model.eval()
        with torch.no_grad():
            model(*inputs)
    finally:
        # Undo
        model.train(restore_training)
        for h in hooks:
            h.remove()

    return make_output(
        summaries=module_summaries,
        batch_size=batch_size,
        input_sizes=input_sizes,
        include_input_shape=include_input_shape,
    )


def make_output(
    summaries: Dict[str, ModuleSummary],
    batch_size: int,
    input_sizes: List[torch.Size],
    include_input_shape: bool = False,
) -> str:

    total_params = 0
    total_output = 0
    total_output_approx = False
    trainable_params = 0
    layer_table = []
    for m in sorted(summaries.values(), key=lambda m: m.sort_key):
        row = [
            c
            for c in [
                m.name,
                (
                    pformat(m.input_shape.dump(), width=SHAPE_COL_WIDTH)
                    if include_input_shape
                    else None
                ),
                pformat(m.output_shape.dump(), width=SHAPE_COL_WIDTH),
                m.num_params,
            ]
            if c is not None
        ]
        layer_table.append(row)

        total_params += m.num_params
        if (output_size := m.output_shape.size) is not None:
            total_output += output_size
        else:
            total_output_approx = True
        if m.trainable:
            trainable_params += m.num_params

    # assume 4 bytes/number (float on cuda).
    bytes_per_mb = 1024 ** 2
    bytes_per_num = 4
    total_input_size = abs(
        np.prod(sum(input_sizes, [])) * batch_size * bytes_per_num / bytes_per_mb
    )
    total_output_size = abs(
        2.0 * total_output * bytes_per_num / bytes_per_mb
    )  # x2 for gradients
    total_params_size = abs(total_params * bytes_per_num / bytes_per_mb)
    total_size = total_params_size + total_output_size + total_input_size

    summary_table = [
        ["Total params", total_params],
        ["Trainable params", trainable_params],
        ["Non-trainable params", total_params - trainable_params],
        ["Input size (MB)", total_input_size],
        ["Forward/backward pass size (MB)", total_output_size],
        ["Params size (MB)", total_params_size],
        [
            f"Estimated Total Size (MB) {'(some layers missing)' if total_output_approx else ''}",
            total_size,
        ],
    ]

    headers = [
        h
        for h in [
            "Layer (type)",
            "Input Shape" if include_input_shape else None,
            "Output Shape",
            "Num params",
        ]
        if h is not None
    ]
    summary_str = "\n".join(
        [
            tabulate(layer_table, headers=headers, floatfmt=",.0f", tablefmt="psql",),
            "",
            tabulate(summary_table, floatfmt=",.1f", tablefmt="psql"),
        ]
    )

    return summary_str
