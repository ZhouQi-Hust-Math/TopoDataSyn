import torch
import warnings
import time
import math

torch.set_default_dtype(torch.float64)  # 精度默认为double类型

class MLP(torch.nn.Module):
    def __init__(self, width=2, depth=2, w_in=2, w_out=2, acf=torch.nn.ELU(), layer=None, batchnorm1d=False):
        super(MLP, self).__init__()
        if layer is None and depth >=1:
            layer = [width for i in range(depth)]
        elif layer is None and depth == 0:
            layer = [width]
        self.net1 = self._make_layer(w_in=w_in, depth=depth, acf=acf, layer=layer, batchnorm1d=batchnorm1d)
        self.net2 = torch.nn.Sequential(
            torch.nn.Linear(layer[-1], w_out),
        )
        if batchnorm1d:
            self.net2.append(torch.nn.BatchNorm1d(w_out, affine=False))
        print('逐层网络已创建, 宽度逐层为%s, 深度为%d, 激活函数列表为%s' % (layer, depth, acf))
        self.width = width
        self.depth = depth

    def _make_layer(self, w_in=2, depth=1, acf=torch.nn.ELU(), layer=None, batchnorm1d=False):
        layers = []
        if isinstance(acf, list) and depth >=1:
            layers = [torch.nn.Linear(w_in, layer[0]), acf[0]]
        elif depth >=1:
            layers = [torch.nn.Linear(w_in, layer[0]), acf]
        if batchnorm1d:
            layers.append(torch.nn.BatchNorm1d(layer[0], affine=False))
        for i in range(len(layer) - 1):  # 创建额外的中间层
            layers.append(torch.nn.Linear(layer[i], layer[i+1]))
            if isinstance(acf, list):
                layers.append(acf[i + 1])
            else:
                layers.append(acf)
            if batchnorm1d:
                layers.append(torch.nn.BatchNorm1d(layer[i+1], affine=False))
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.net1(x)
        x = self.net2(x)
        return x

class MLPpre(MLP):
    def __init__(self, width=2, depth=2, w_in=2, w_out=2, acf=torch.nn.ELU(), layer=None):
        super(MLPpre, self).__init__(width=width, depth=depth, w_in=w_in, w_out=w_out, acf=acf, layer=layer)

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

class building_resblock(torch.nn.Module):
    def __init__(self, width=3, blockwidth=1, acf=[torch.nn.ELU()], batchnorm1d=False):
        super(building_resblock, self).__init__()
        self.model = self._make_block(width=width, blockwidth=blockwidth, acf=acf, batchnorm1d=batchnorm1d)

    def _make_block(self, width=2, blockwidth=1, acf=[torch.nn.ELU()], batchnorm1d=False):
        if not isinstance(acf, list):
            acf = [acf]
        layers = [torch.nn.Linear(width, blockwidth), acf[0]]
        if batchnorm1d:
            layers.append(torch.nn.BatchNorm1d(blockwidth, affine=False))

        for i in range(len(acf) - 1):  # 创建额外的中间层
            layers.append(torch.nn.Linear(blockwidth, blockwidth))
            layers.append(acf[i + 1])
            if batchnorm1d:
                layers.append(torch.nn.BatchNorm1d(blockwidth, affine=False))

        layers.append(torch.nn.Linear(blockwidth, width))
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        down = x
        result = self.model(x)
        # 残差相加
        result += down
        return result

class ResNet(torch.nn.Module):
    def __init__(self, width=3, w_in=3, w_out=2, acf=[torch.nn.ELU()], blocknum=1, blockwidth=1, blockdepth=2, batchnorm1d=False):
        super(ResNet, self).__init__()
        if isinstance(acf, list):
            acf=acf
        else:
            acf=[acf for k in range(blocknum*blockdepth)]
        assert len(acf) == blocknum*blockdepth and blocknum>=1 and blockdepth>=1
        self.net1 = torch.nn.Sequential(
            torch.nn.Linear(w_in, width),
        )
        self.blocks = self._make_layer(width=width, acf=acf, blockwidth=blockwidth, blocknum=blocknum, blockdepth=blockdepth,
                                       batchnorm1d=batchnorm1d)
        self.net2 = torch.nn.Sequential(
            torch.nn.Linear(width, w_out),
        )
        if batchnorm1d:
            self.net2.append(torch.nn.BatchNorm1d(w_out, affine=False))

    def _make_layer(self, width=3, acf=[torch.nn.ELU()], blocknum=1, blockwidth=1, blockdepth=2, batchnorm1d=False):
        layers = []
        for i in range(blocknum):
            layers.append(building_resblock(width=width, blockwidth=blockwidth,
                                            acf=acf[blockdepth * i : blockdepth * (i+1)], batchnorm1d=batchnorm1d))
            if i != blocknum-1:
                layers.append(torch.nn.Linear(width, width))
        return torch.nn.Sequential(*layers)  # 将列表解码

    def forward(self, x):
        x = self.net1(x)
        x = self.blocks(x)
        x = self.net2(x)
        return x


date_str = '26-03-11'
if __name__ == '__main__':
    print('最新更改日期：%s' % date_str)
    print('作者：周琦')
    print('联系方式：2517036487@qq.com')
else:
    print('已导入模型', __name__, '最新更改日期:', date_str)
