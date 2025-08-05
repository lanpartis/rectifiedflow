import torch
from torch import nn


class FlowMatching:
    def __init__(self, model: nn.Module):
        self.model = model
        self.device = next(model.parameters()).device

    @staticmethod
    def get_train_tuple(z1, z0=None, repeat=1):
        bs, *other_dim = z1.shape
        t = torch.rand((z1.shape[0], repeat, 1), device=z1.device)
        z1 = z1.reshape(bs, 1, -1).expand(-1, repeat, -1)
        if not z0 is None:
            z0 = z0.reshape(bs, 1, -1).expand(-1, repeat, -1)
        else:
            z0 = torch.randn_like(z1)
        z_t = t * z1 + (1.0 - t) * z0
        target = z1 - z0
        t = t.reshape(-1)
        z_t = z_t.reshape(-1, *other_dim)
        target = target.reshape(-1, *other_dim)
        return z_t, t, target

    @torch.no_grad()
    def sample_ode(self, z0, N=100):
        dt = 1.0 / N
        traj = []
        z = z0.detach().clone()
        batchsize = z.shape[0]
        traj.append(z.detach().clone())
        for i in range(N):
            t = torch.ones((batchsize), device=self.device) * i / N
            pred = self.model(z, t)
            z = z.detach().clone() + pred * dt
            traj.append(z.detach().clone())
        return traj


if __name__ == "__main__":
    sample = torch.range(1, 48).reshape(2, 4, 3, 2, 1)
    zt, t, tar = FlowMatching.get_train_tuple(sample[0], sample[1], 2)
    print(sample.shape)
    print(zt.shape, t.shape, tar.shape)
