
import torch
from utillc import *
print_everything()

N = 7

todev = lambda x : x.cuda()

todev = lambda x : x
T= torch.float32

fu = torch.nn.Conv2d(1, 1, (2,1), padding='same')
fu.weight.data=torch.tensor([[[[-1.], [1.]]]])
fu.bias.data = torch.zeros((1))

right = torch.nn.Conv2d(1, 1, (2,1), padding='same')
right.weight.data=torch.tensor([[[[0.], [1.]]]])
right.bias.data = torch.zeros((1))

fv = torch.nn.Conv2d(1, 1, (1,2), padding='same')
fv.weight.data = torch.tensor([[[[1., -1.]]]])
fv.bias.data = torch.zeros((1))

up = torch.nn.Conv2d(1, 1, (1,2), padding='same')
up.weight.data = torch.tensor([[[[1., 0.]]]])
up.bias.data = torch.zeros((1))

u = torch.zeros((1, N, N), dtype=T)
v = torch.zeros((1, N, N), dtype=T)

u[0, 2, 2] = 3
v[0, 2, 3] = 5

EKON(u.shape)
d = fu(u) + fv(v)
EKON(d.shape)
EKOX(d.detach().numpy())

fs = torch.nn.Conv2d(1, 1, (3, 3), padding='same')
EKOX(fs.weight)
fs.weight.data = torch.tensor([[[[ 0.,  1.,  0.],
								 [ 1.,  0.,  1.],
								 [ 0.,  1.,  0.]]]])

fs.bias.data = torch.zeros((1))
s = torch.zeros((1, N,N), dtype=T)
s[0,3,3] = 1

ss = fs(s)
u +=  d * up(s) / ss
u -=  d * s / ss
v +=  d * right(s) / ss
v -=  d * s /ss
EKOX(ss)
EKOX(ss.shape)








