import torch
import numpy as np

from tabulate import tabulate
import attr

from typing import List, Dict, Callable, Any, Iterable, Optional, Union
from functools import partial, cached_property
from pprint import pformat


gpu_if_available = "cuda:0" if torch.cuda.is_available() else "cpu"


def get_tensor_size(input: torch.Tensor, idx: int) -> torch.Size:
    return input.size()


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
):
    result, params_info = summary_string(
        model, *inputs, batch_size=batch_size, get_input_size=get_input_size
    )
    print(result)

    return params_info


def shape(arg, maxlen=3) -> Shape:
    return Shape(arg, maxlen=maxlen)


SHAPE_COL_WIDTH = 50


@attr.s
class ModuleSummary:
    key: str = attr.ib()
    input_shape: Shape = attr.ib(init=False)
    output_shape: Shape = attr.ib(init=False)
    trainable: bool = attr.ib(init=False, default=False)
    num_params: int = attr.ib(init=False, default=0)


def register_hook(
    module, bs: int, summaries: Dict[str, ModuleSummary], reg_hooks: List
) -> None:
    def hook(module, input, output):
        class_name = str(module.__class__).split(".")[-1].split("'")[0]
        module_idx = len(summaries)

        msum = ModuleSummary(f"{class_name}-{module_idx+1}")
        # Input is a tuple of size 1, always?
        msum.input_shape = shape(*input)
        msum.output_shape = shape(output)

        if hasattr(module, "weight") and hasattr(module.weight, "size"):
            msum.num_params += np.prod(module.weight.size())
            msum.trainable = module.weight.requires_grad
        if hasattr(module, "bias") and hasattr(module.bias, "size"):
            msum.num_params += np.prod(module.bias.size())

        summaries[msum.key] = msum

    if not isinstance(module, torch.nn.Sequential) and not isinstance(
        module, torch.nn.ModuleList
    ):
        reg_hooks.append(module.register_forward_hook(hook))


def make_random_input(
    *shape: int, dtype=torch.FloatTensor, device=gpu_if_available, bs: int = 2
):
    return torch.rand(bs, *shape).type(dtype).to(device=device)


def summary_string(
    model: torch.nn.Module,
    *inputs: Any,
    batch_size: int = -1,
    get_input_size: Callable[[Any, int], torch.Size] = get_tensor_size,
):
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
        # register hook
        model.apply(
            partial(
                register_hook,
                reg_hooks=hooks,
                summaries=module_summaries,
                bs=batch_size,
            )
        )

        # make a forward pass
        model.eval()
        with torch.no_grad():
            model(*inputs)
    finally:
        # Undo
        model.train(restore_training)
        for h in hooks:
            h.remove()

    total_params = 0
    total_output = 0
    total_output_approx = False
    trainable_params = 0
    layer_table = []
    for layer, msum in module_summaries.items():
        layer_table.append(
            [
                layer,
                pformat(msum.output_shape.dump(), width=SHAPE_COL_WIDTH),
                msum.num_params,
            ]
        )
        total_params += msum.num_params
        if (output_size := msum.output_shape.size) is not None:
            total_output += output_size
        else:
            total_output_approx = True
        if msum.trainable:
            trainable_params += msum.num_params

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

    summary_str = "\n".join(
        [
            tabulate(
                layer_table,
                headers=["Layer (type)", "Output Shape", "Param #"],
                floatfmt=",.0f",
                tablefmt="psql",
            ),
            "",
            tabulate(summary_table, floatfmt=",.1f", tablefmt="psql"),
        ]
    )

    return summary_str, (total_params, trainable_params)
