import torch
import numpy as np

from tabulate import tabulate
import attr

from typing import List, Dict, Callable, Any, Iterable
from functools import partial

def get_tensor_sizes(inputs: Iterable[torch.Tensor]) -> List[torch.Size]:
    return [i.size() for i in inputs]

def summary(model: torch.nn.Module, *inputs: Any, batch_size: int=-1, get_input_sizes: Callable[[Iterable[Any]], List[torch.Size]] = get_tensor_sizes):
    result, params_info = summary_string(model, *inputs, batch_size=batch_size, get_input_sizes=get_input_sizes)
    print(result)

    return params_info

@attr.s
class ModuleSummary:
    key: str = attr.ib()
    input_shape: List[int] = attr.ib(init=False)
    output_shape: List[int] = attr.ib(init=False)
    trainable: bool = attr.ib(init=False, default=False)
    num_params: int = attr.ib(init=False, default=0)

def register_hook(module, bs: int, module_summaries: Dict[str, ModuleSummary], reg_hooks: List) -> None:
    def hook(module, input, output):
        class_name = str(module.__class__).split(".")[-1].split("'")[0]
        module_idx = len(module_summaries)

        msum = ModuleSummary(f'{class_name}-{module_idx+1}')
        msum.input_shape = [bs] + list(input[0].size())[1:]
        msum.output_shape = [
            [-1] + list(o.size())[1:] for o in output
        ] if isinstance(output, (list, tuple)) else ([bs] + list(output.size())[1:])

        if hasattr(module, "weight") and hasattr(module.weight, "size"):
            msum.num_params += np.prod(module.weight.size())
            msum.trainable = module.weight.requires_grad
        if hasattr(module, "bias") and hasattr(module.bias, "size"):
            msum.num_params += np.prod(module.bias.size())

        module_summaries[msum.key] = msum

    if (
        not isinstance(module, torch.nn.Sequential)
        and not isinstance(module, torch.nn.ModuleList)
    ):
        reg_hooks.append(module.register_forward_hook(hook))

def make_random_input(*shape: int, dtype=torch.FloatTensor, device=torch.device('cuda:0'), bs: int = 2):
    return torch.rand(bs, *shape).type(dtype).to(device=device)


def summary_string(model: torch.nn.Module, *inputs: Any, batch_size: int=-1, get_input_sizes: Callable[[Iterable[Any]], List[torch.Size]] = get_tensor_sizes):
    # if dtypes == None:
    #     dtypes = [torch.FloatTensor]*len(input_size)

    # # multiple inputs to the network
    # if isinstance(input_size, tuple):
    #     input_size = [input_size]

    # # batch_size of 2 for batchnorm
    # x = [make_random_input(*in_size, dtype=dtype, device=device, bs=2) for in_size, dtype in zip(input_size, dtypes)]

    input_sizes = get_input_sizes(inputs)
    assert len(input_sizes) == len(inputs), f"Input sizes from get_input_sizes must match the number of inputs: {len(inputs)=} {len(input_sizes)=}"
    assert all(sz[0] >= 2 for sz in input_sizes), "Each input must have batch_size >= 2 (for batchnorm)"

    # create properties
    summ: Dict[str, ModuleSummary] = {}
    hooks: List = []

    # register hook
    model.apply(partial(register_hook, reg_hooks=hooks, module_summaries=summ, bs=batch_size))

    # make a forward pass
    # print(x.shape)
    model(*inputs)

    # remove these hooks
    for h in hooks:
        h.remove()

    total_params = 0
    total_output = 0
    trainable_params = 0
    layer_table = []
    for layer, msum in summ.items():
        layer_table.append([layer, msum.output_shape, msum.num_params])
        total_params += msum.num_params

        total_output += np.prod(msum.output_shape)
        if msum.trainable:
            trainable_params += msum.num_params

    # assume 4 bytes/number (float on cuda).
    bytes_per_mb = 1024 ** 2
    bytes_per_num = 4
    total_input_size = abs(np.prod(sum(input_sizes, ())) * batch_size * bytes_per_num / bytes_per_mb)
    total_output_size = abs(2. * total_output * bytes_per_num / bytes_per_mb)  # x2 for gradients
    total_params_size = abs(total_params * bytes_per_num / bytes_per_mb)
    total_size = total_params_size + total_output_size + total_input_size

    summary_table = [
        ["Total params", total_params],
        ["Trainable params", trainable_params],
        ["Non-trainable params", total_params - trainable_params],
        ["Input size (MB)", total_input_size],
        ["Forward/backward pass size (MB)", total_output_size],
        ["Params size (MB)", total_params_size],
        ["Estimated Total Size (MB)", total_size]
    ]

    summary_str = "\n".join([
        tabulate(layer_table, headers=["Layer (type)", "Output Shape", "Param #"]),
        tabulate(summary_table)
    ])

    # return summ
    return summary_str, (total_params, trainable_params)
