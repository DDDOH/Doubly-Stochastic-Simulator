import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, seed_dim, hidden_size, n_interval):
        super(Generator, self).__init__()
        n_layer = len(hidden_size)
        self.normal = torch.distributions.normal.Normal(
            loc=0, scale=1, validate_args=None)

        nonlinear_layers = [
            nn.Linear(seed_dim, hidden_size[0]), nn.LeakyReLU(0.1, True)]
        for i in range(n_layer-1):
            nonlinear_layers.append(nn.Linear(hidden_size[i], hidden_size[i + 1]))
            nonlinear_layers.append(nn.LeakyReLU(0.1, True))
        nonlinear_layers.append(nn.Linear(hidden_size[-1], n_interval))

        self.nonlinear = nn.Sequential(*nonlinear_layers)
        # self.linear = nn.Linear(seed_dim, n_interval)

    def forward(self, noise):
        return self.nonlinear(noise)


class DSSimulator(Generator):
    def __init__(self, seed_dim, hidden_size, n_interval):
        Generator.__init__(self, seed_dim, hidden_size, n_interval)
        self.poisson_simulator = PoissonSimulator()

    def forward(self, noise, return_intensity=False):
        intensity = self.nonlinear(noise)
        count = self.poisson_simulator.apply(intensity)
        if return_intensity:
            return count, intensity
        else:
            return count


class PoissonSimulator(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        sign = torch.sign(input)
        "The operator 'aten::poisson' is not current implemented for the MPS device. If you want this op to be added in priority during the prototype phase of this feature, please comment on https://github.com/pytorch/pytorch/issues/77764. As a temporary fix, you can set the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1` to use the CPU as a fallback for this op. WARNING: this will be slower than running natively on MPS."
        X = torch.poisson(torch.abs(input))
        X = sign * X
        ctx.save_for_backward(input, X)
        return X

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, X = ctx.saved_tensors
        return torch.clip(1 + (X - input) / (2 * input), 0.5, 1.5) * grad_output


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    # possion_simulator = PoissonSimulator()
    # lam = torch.arange(-1000, 1000, 1, dtype=torch.float32, requires_grad=True)
    # X_np = np.sign(lam.detach()) * np.random.poisson(np.abs(lam.detach()))
    # X_torch = possion_simulator.apply(lam)
    # X_torch.sum().backward()
    # plt.figure(figsize=(10, 4))
    # plt.subplot(121)
    # plt.plot(lam.detach().numpy(), X_torch.detach(), label='torch')
    # plt.plot(lam.detach().numpy(), X_np, label='numpy')
    # plt.legend()

    # plt.subplot(122)
    # plt.plot(lam.detach(), lam.grad.detach().numpy(), label='grad')
    # plt.show()

    seed_dim = 5
    simulator = DSSimulator(seed_dim=seed_dim, dim=[64, 64, 64], n_interval=10)
    batch_size = 100
    noise = torch.randn(batch_size, seed_dim)
    count = simulator.forward(noise)
