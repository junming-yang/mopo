# implement model to learn state transitions and rewards
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
from common import util


def get_network(param_shape, deconv=False):
    """
    Parameters
    ----------
    param_shape: tuple, length:[(4, ), (2, )], optional

    deconv: boolean
        Only work when len(param_shape) == 4.
    """

    if len(param_shape) == 4:
        if deconv:
            in_channel, kernel_size, stride, out_channel = param_shape
            return torch.nn.ConvTranspose2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride)
        else:
            in_channel, kernel_size, stride, out_channel = param_shape
            return torch.nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride)
    elif len(param_shape) == 2:
        in_dim, out_dim = param_shape
        return torch.nn.Linear(in_dim, out_dim)
    else:
        raise ValueError(f"Network shape {param_shape} illegal.")


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x


def get_act_cls(act_fn_name):
    act_fn_name = act_fn_name.lower()
    if act_fn_name == "tanh":
        act_cls = torch.nn.Tanh
    elif act_fn_name == "sigmoid":
        act_cls = torch.nn.Sigmoid
    elif act_fn_name == 'relu':
        act_cls = torch.nn.ReLU
    elif act_fn_name == 'identity':
        act_cls = torch.nn.Identity
    elif act_fn_name == 'swish':
        act_cls = Swish
    else:
        raise NotImplementedError(f"Activation functtion {act_fn_name} is not implemented. \
            Possible choice: ['tanh', 'sigmoid', 'relu', 'identity'].")
    return act_cls


class MLPNetwork(nn.Module):
    def __init__(
            self, input_dim: int,
            out_dim: int,
            hidden_dims: Union[int, list],
            act_fn="relu",
            out_act_fn="identity",
            **kwargs
    ):
        super(MLPNetwork, self).__init__()
        if len(kwargs.keys()) > 0:
            warn_str = "Redundant parameters for MLP network {}.".format(kwargs)
            warnings.warn(warn_str)

        if type(hidden_dims) == int:
            hidden_dims = [hidden_dims]
        hidden_dims = [input_dim] + hidden_dims
        self.networks = []
        act_cls = get_act_cls(act_fn)
        out_act_cls = get_act_cls(out_act_fn)

        for i in range(len(hidden_dims) - 1):
            curr_shape, next_shape = hidden_dims[i], hidden_dims[i + 1]
            curr_network = get_network([curr_shape, next_shape])
            self.networks.extend([curr_network, act_cls()])
        final_network = get_network([hidden_dims[-1], out_dim])
        self.networks.extend([final_network, out_act_cls()])
        self.networks = nn.Sequential(*self.networks)

    def forward(self, input):
        return self.networks(input)

    @property
    def weights(self):
        return [net.weight for net in self.networks if isinstance(net, torch.nn.modules.linear.Linear)]


class EnsembleModel(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dims, device, ensemble_size=7, num_elite=5, decay_weights=None,
                 act_fn="swish", out_act_fn="identity", reward_dim=1, **kwargs):
        super(EnsembleModel, self).__init__()
        assert (decay_weights is None or len(decay_weights) == len(hidden_dims) + 1)
        self.out_dim = obs_dim + reward_dim

        self.ensemble_models = [
            MLPNetwork(input_dim=obs_dim + action_dim, out_dim=self.out_dim * 2, hidden_dims=hidden_dims, act_fn=act_fn,
                       out_act_fn=out_act_fn) for _ in range(ensemble_size)]
        for i in range(ensemble_size):
            self.add_module("model_{}".format(i), self.ensemble_models[i])

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_elite = num_elite
        self.ensemble_size = ensemble_size
        self.decay_weights = decay_weights
        self.elite_model_idxes = torch.tensor([i for i in range(num_elite)])
        self.max_logvar = nn.Parameter((torch.ones((1, self.out_dim)).float() / 2).to(device), requires_grad=True)
        self.min_logvar = nn.Parameter((-torch.ones((1, self.out_dim)).float() * 10).to(device), requires_grad=True)
        self.register_parameter("max_logvar", self.max_logvar)
        self.register_parameter("min_logvar", self.min_logvar)
        self.to(device)

    def predict(self, input):
        # convert input to tensors
        if type(input) != torch.Tensor:
            if len(input.shape) == 1:
                input = torch.FloatTensor([input]).to(util.device)
            else:
                input = torch.FloatTensor(input).to(util.device)
        # predict
        if len(input.shape) == 3:
            model_outputs = [net(ip) for ip, net in zip(torch.unbind(input), self.ensemble_models)]
        elif len(input.shape) == 2:
            model_outputs = [net(input) for net in self.ensemble_models]
        predictions = torch.stack(model_outputs)

        mean = predictions[:, :, :self.out_dim]
        logvar = predictions[:, :, self.out_dim:]
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        return mean, logvar

    def get_decay_loss(self):
        decay_losses = []
        for model_net in self.ensemble_models:
            curr_net_decay_losses = [decay_weight * torch.sum(torch.square(weight)) for decay_weight, weight in
                                     zip(self.decay_weights, model_net.weights)]
            decay_losses.append(torch.sum(torch.stack(curr_net_decay_losses)))
        return torch.sum(torch.stack(decay_losses))

    def load_state_dicts(self, state_dicts):
        for i in range(self.ensemble_size):
            self.ensemble_models[i].load_state_dict(state_dicts[i])


if __name__ == "__main__":
    device = torch.device("cpu")
    model = EnsembleModel(10, 3, [20, 20], device)
    for p in model.parameters():
        print(p)
