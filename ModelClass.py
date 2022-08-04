import torch
import torch.nn as nn


class Swish(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class Gaussian(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, x):
        return torch.exp(-5 * x ** 2)


class Sin(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)


class Snake(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = 0.5

    def forward(self, x):
        return x + torch.sin(self.alpha * x) ** 2 / self.alpha


def activation(name):
    if name in ['tanh', 'Tanh']:
        return nn.Tanh()
    elif name in ['relu', 'ReLU']:
        return nn.ReLU(inplace=True)
    elif name in ['lrelu', 'LReLU']:
        return nn.LeakyReLU(inplace=True)
    elif name in ['sigmoid', 'Sigmoid']:
        return nn.Sigmoid()
    elif name in ['softplus', 'Softplus']:
        return nn.Softplus(beta=4)
    elif name in ['celu', 'CeLU']:
        return nn.CELU()
    elif name in ['sin', 'Sin']:
        return Sin()
    elif name in ['swish']:
        return Swish()
    elif name in ['snake']:
        return Snake()
    elif name in ['gaussian']:
        return Gaussian()
    else:
        raise ValueError('Unknown activation function')


class Pinns(nn.Module):

    def __init__(self, input_dimension, output_dimension, network_properties):
        super(Pinns, self).__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.n_hidden_layers = int(network_properties["hidden_layers"])
        self.neurons = int(network_properties["neurons"])
        self.lambda_residual = float(network_properties["residual_parameter"])
        self.kernel_regularizer = int(network_properties["kernel_regularizer"])
        self.regularization_param = float(network_properties["regularization_parameter"])
        self.num_epochs = int(network_properties["epochs"])
        self.act_string = str(network_properties["activation"])
        self.iterations = int(network_properties["iterations"])
        self.reset_freq = network_properties["reset_freq"]
        self.loss = network_properties["loss_type"]

        self.activation = activation(self.act_string)

        if self.n_hidden_layers != 0:
            self.input_layer = nn.Linear(self.input_dimension, self.neurons)
            self.hidden_layers = nn.ModuleList([nn.Linear(self.neurons, self.neurons) for _ in range(self.n_hidden_layers - 1)])
            self.output_layer = nn.Linear(self.neurons, self.output_dimension)
        else:
            self.input_output_layer = nn.Linear(self.input_dimension, self.output_dimension)

    def forward(self, x):
        if self.n_hidden_layers != 0:
            x = self.activation(self.input_layer(x))
            for k, l in enumerate(self.hidden_layers):
                x = self.activation(l(x))
            return self.output_layer(x)
        else:
            return self.input_output_layer(x)


class PinnsTest(nn.Module):

    def __init__(self, input_dimension, output_dimension, network_properties):
        super(PinnsTest, self).__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.n_hidden_layers = int(network_properties["hidden_layers"])
        self.neurons = int(network_properties["neurons"])
        self.lambda_residual = float(network_properties["residual_parameter"])
        self.kernel_regularizer = int(network_properties["kernel_regularizer"])
        self.regularization_param = float(network_properties["regularization_parameter"])
        self.num_epochs = int(network_properties["epochs"])
        self.act_string = str(network_properties["activation"])
        self.iterations = int(network_properties["iterations"])
        self.reset_freq = (network_properties["reset_freq"])

        self.input_layer = nn.Linear(self.input_dimension, self.neurons)
        self.hidden_layers = nn.ModuleList([nn.Linear(self.neurons, self.neurons) for _ in range(self.n_hidden_layers - 1)])
        self.output_layer = nn.Linear(self.neurons, self.output_dimension)

        self.activation = activation(self.act_string)
        self.activation_out = Gaussian()
        self.activation_out_2 = torch.nn.Softplus()

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for k, l in enumerate(self.hidden_layers):
            x = self.activation(l(x))
        o = self.output_layer(x)
        return o
