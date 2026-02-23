import torch
from TopoDataSyn import Plot_command
from TopoDataSyn.NN_model import Netpre

nets = Netpre(width=3, w_in=2, w_out=2, depth=4, acf=torch.nn.ELU())
print(nets)
nets.net1 = nets.net1[0: 8]
print(nets)
