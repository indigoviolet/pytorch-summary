from ..torchsummary import summary, summary_string, make_random_input
from .example_models import (
    SingleInputNet,
    MultipleInputNet,
    MultipleInputNetDifferentDtypes,
)
import torch

gpu_if_available = "cuda:0" if torch.cuda.is_available() else "cpu"


def assertEqual(a, b):
    assert a == b


class TestSummary:
    def test_single_input(self):
        model = SingleInputNet()
        input = make_random_input(1, 28, 28, device="cpu")
        total_params, trainable_params = summary(model, input)
        assertEqual(total_params, 21840)
        assertEqual(trainable_params, 21840)

    def test_multiple_input(self):
        model = MultipleInputNet()
        input1 = make_random_input(1, 300, device="cpu")
        input2 = make_random_input(1, 300, device="cpu")
        total_params, trainable_params = summary(model, input1, input2)
        assertEqual(total_params, 31120)
        assertEqual(trainable_params, 31120)

    def test_single_layer_network(self):
        model = torch.nn.Linear(2, 5)
        input = make_random_input(1, 2, device="cpu")
        total_params, trainable_params = summary(model, input)
        assertEqual(total_params, 15)
        assertEqual(trainable_params, 15)

    def test_single_layer_network_on_gpu(self):
        model = torch.nn.Linear(2, 5)
        if torch.cuda.is_available():
            model.cuda()
            input = make_random_input(1, 2, device=gpu_if_available)
            total_params, trainable_params = summary(model, input)
            assertEqual(total_params, 15)
            assertEqual(trainable_params, 15)

    def test_multiple_input_types(self):
        model = MultipleInputNetDifferentDtypes()
        input1 = make_random_input(1, 300, device="cpu")
        input2 = make_random_input(1, 300, device="cpu", dtype=torch.LongTensor)
        total_params, trainable_params = summary(model, input1, input2)
        assertEqual(total_params, 31120)
        assertEqual(trainable_params, 31120)


class TestSummaryString:
    def test_single_input(self):
        model = SingleInputNet()
        input = make_random_input(1, 28, 28, device="cpu")
        result, (total_params, trainable_params) = summary_string(model, input)
        print(result)
        assertEqual(type(result), str)
        assertEqual(total_params, 21840)
        assertEqual(trainable_params, 21840)
