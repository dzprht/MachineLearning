from torch import nn

from typing import TypeAlias, Literal


ActivationFunction: TypeAlias = Literal["relu", "prelu", "sigmoid", "softmax"]

def activations_func(activation_name: ActivationFunction | nn.Module) -> nn.Module:
    act_layers = {
        "relu": nn.ReLU(inplace=True),
        "sigmoid": nn.Sigmoid(),
        "prelu": nn.PReLU(),
        "softmax": nn.Softmax(),
    }

    if isinstance(activation_name, ActivationFunction):
        act_layer = act_layers[activation_name.lower()]
    elif issubclass(activation_name, nn.Module):
        act_layer = activation_name
    else:
        raise NotImplementedError()

    return act_layer
    