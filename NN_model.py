import numpy as np
import torch
import warnings
import time
import math

torch.set_default_dtype(torch.float64)  # 精度默认为double类型

class Net(torch.nn.Module):
    def __init__(self, width=2, depth=2, w_in=2, w_out=2, acf=torch.nn.ELU(), layer=None):
        super(Net, self).__init__()
        if layer is None:
            if not (isinstance(depth, int) and depth > 0):
                warnings.warn('深度不是正整数')
            self.net1 = self._make_layer(width=width, depth=depth, w_in=w_in, acf=acf)
            self.net2 = torch.nn.Sequential(
                torch.nn.Linear(width, w_out),
            )
            print('网络已创建，宽度为%d，深度为%d，激活函数为%s' % (width, depth, acf))
            self.width = width
            self.depth = depth
        else:
            depth = len(layer)
            self.net1 = self._make_layerwise(w_in=w_in, acf=acf, layer=layer)
            self.net2 = torch.nn.Sequential(
                torch.nn.Linear(layer[-1], w_out),
                )
            print('逐层网络已创建, 宽度逐层为%s, 深度为%d, 激活函数为%s' % (layer, depth, acf))


    def _make_layer(self, width=2, depth=2, w_in=2, acf=torch.nn.ELU()):
        layers = [torch.nn.Linear(w_in, width), acf]
        for i in range(depth - 1):  # 创建额外的中间层
            layers.append(torch.nn.Linear(width, width))
            layers.append(acf)
        return torch.nn.Sequential(*layers)

    def _make_layerwise(self, w_in=2, acf=torch.nn.ELU(), layer=None):
        layers = [torch.nn.Linear(w_in, layer[0]), acf]
        for i in range(len(layer) - 1):  # 创建额外的中间层
            layers.append(torch.nn.Linear(layer[i], layer[i+1]))
            layers.append(acf)
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.net1(x)
        x = self.net2(x)
        return x

class Netpre(Net):
    def __init__(self, width=2, depth=2, w_in=2, w_out=2, acf=torch.nn.ELU(), layer=None):
        super(Netpre, self).__init__(width=width, depth=depth, w_in=w_in, w_out=w_out, acf=acf, layer=layer)

    def forward(self, x):
        x = self.net1(x)
        return x

def retrain(model=None, dataloader=None, lr=1e-4, newpath=None, device='cuda:0', iteration=1000000, frameduration=1000):
    model.to(device)
    start_time = time.time()
    opt_s = torch.optim.Adam(params=model.parameters(), lr=lr)
    loss = torch.nn.MSELoss(reduction='mean')

    for idx, (Input, output) in enumerate(dataloader):  # 对数据集分批读取
        Input = Input.to(device)
        output = output.to(device)
        for t in range(iteration):
            if (t + 1) % frameduration == 0:
                opt_s.zero_grad()
                LOSS = loss(model(Input), output)
                LOSS.backward()
                opt_s.step()
                record_time = time.time()
                remain_time = (record_time - start_time) * (iteration - t - 1) / (t + 1)
                m, s = divmod(remain_time, 60)
                h, m = divmod(m, 60)
                print('迭代次数', t + 1, '进度%.2f%%' % (100 * (t+1)/iteration), 'loss=', LOSS.item(), 'lr=', lr,
                      '预计剩余时间%d小时%d分%.0f秒' % (h, m, s))
            else:
                opt_s.zero_grad()
                LOSS = loss(model(Input), output)
                LOSS.backward()
                opt_s.step()
    net_para = {'net': model.state_dict()}
    print('新模型参数已保存至%s' % newpath)
    torch.save(net_para, '%s' % newpath)


def tensor_select_index(trainset=None, testset=None):
    index_list = []
    _, indices = torch.sort(trainset[:, 0], dim=0, descending=False)
    trainset = trainset[indices, :]

    key_list = [k for k in range(math.floor(10 * (trainset[-1, 0] - 0.001)), math.floor(10 * (trainset[0, 0] - 0.001))-1, -1)]

    range_dict = {}
    temp = trainset.size()[0] - 1
    for key in key_list:
        for i in range(temp, -1, -1):
            if math.floor(10 * (trainset[i, 0] - 0.001)) >= key:
                range_dict['%s' % str(key)] = i
            elif math.floor(10 * (trainset[i+1, 0] - 0.001)) >= key:
                range_dict['%s' % str(key)] = i + 1
                temp = i
                break

    for i in range(testset.size()[0]):
        dif_judge = True
        key = math.floor(10 * (testset[i, 0] - 1e-5))

        if key <= key_list[-1]:
            a = 0
            b = range_dict['%s' % str(key_list[-1])]

        elif key >= key_list[0]:
            a = range_dict['%s' % str(key_list[0])]
            b = trainset.size()[0]-1

        else:
            a = range_dict['%s' % str(key-1)]
            b = range_dict['%s' % str(key+1)]

        for j in range(a, b+1, 1):
            if torch.allclose(trainset[j, :], testset[i, :], rtol=1e-5):  # 浮点数不能直接用"相等"来判断，存在浮点误差
                dif_judge = False
                print('出现相同元素')
                break
        if dif_judge:
            index_list.append(i)
        print("数据筛选进度%.2f%%" % (100 * (i+1)/testset.size()[0]))
    return index_list

date_str = '26-02-01'
if __name__ == '__main__':
    print('最新更改日期：%s' % date_str)
    print('作者：周琦')
    print('联系方式：2517036487@qq.com')
else:
    print('已导入模型', __name__, '最新更改日期:', date_str)
