import torch
from torch import nn


class RectifiedFlow():
  def __init__(self, model:nn.Module, num_steps=1000):
    self.model = model
    self.N = num_steps
    self.device = next(model.parameters()).device
  
  @staticmethod
  def get_train_tuple(z1, z0=None, repeat=1):

    bs, *other_dim = z1.shape
    t = torch.rand((z1.shape[0],repeat,1),device=z1.device)
    z1 = z1.reshape(bs,1,-1).expand(-1,repeat,-1)
    if not z0 is None:
    #   pass
      z0 = z0.reshape(bs,1,-1).expand(-1,repeat,-1)
    else:
      z0 = torch.randn_like(z1)

    # t = torch.rand((z1.shape[0],1,1,1),device=z1.device)



    z_t =  t * z1 + (1.-t) * z0
    target = z1 - z0 

    t = t.reshape(-1)
    z_t = z_t.reshape(-1,*other_dim)
    target = target.reshape(-1, *other_dim)
    return z_t, t, target

  @torch.no_grad()
  def sample_ode(self, z0=None, N=None):
    '''
    z0第一维是batchsize
    '''
    if N is None:
      N = self.N    
    dt = 1./N
    traj = [] # to store the trajectory
    z = z0.detach().clone()
    batchsize = z.shape[0]
    
    traj.append(z.detach().clone())
    for i in range(N):
      t = torch.ones((batchsize), device=self.device) * i / N
      pred = self.model(z, t)
      z = z.detach().clone() + pred * dt
      
      traj.append(z.detach().clone())

    return traj
  
if __name__ == '__main__':
  sample = torch.range(1,48).reshape(2,4,3,2,1)
  zt,t,tar = RectifiedFlow.get_train_tuple(sample[0],sample[1],2)
  print(sample.shape)
  print(zt.shape,t.shape,tar.shape)