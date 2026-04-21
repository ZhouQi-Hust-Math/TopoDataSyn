import torch
from TopoDataSyn import version_register

class version_info(version_register):
    def __init__(self):
        super().__init__(timeversion='260330-21:56')

# 自定义带可学习参数的激活函数 (已修正权重范围限制)
class PELU(torch.nn.Module):
    def __init__(
            self, num_parameters: int = 1, init: float = 1.0, device=None, dtype=None
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        self.num_parameters = num_parameters
        super().__init__()
        # 可学习参数 a
        self.init = init
        self.weight = torch.nn.Parameter(torch.empty(num_parameters, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Resets parameters based on their initialization used in ``__init__``.
        """
        torch.nn.init.constant_(self.weight, self.init)

    def forward(self, x):
        # 强制 weight 为非负数
        with torch.no_grad():
            self.weight.clamp_(min=0)
            
        if self.num_parameters == 1:
            return torch.where(x >= 0, x, self.weight * (torch.exp(x) - 1))
        else:
            # Broadcast weight over the second dimension (channel dimension)
            # x is expected to have shape (N, C, ...) where C == num_parameters
            shape = [1, self.num_parameters] + [1] * (x.dim() - 2)
            return torch.where(x >= 0, x, self.weight.view(*shape) * (torch.exp(x) - 1))


class PLeakyReLU(torch.nn.Module):
    def __init__(
            self, num_parameters: int = 1, init: float = 1.0, device=None, dtype=None
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        self.num_parameters = num_parameters
        super().__init__()
        # 可学习参数 a
        self.init = init
        self.weight = torch.nn.Parameter(torch.empty(num_parameters, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Resets parameters based on their initialization used in ``__init__``.
        """
        torch.nn.init.constant_(self.weight, self.init)

    def forward(self, x):
        # 强制 weight 在 (0, 1) 之间
        with torch.no_grad():
            self.weight.clamp_(min=1e-4, max=1-1e-4)

        if self.num_parameters == 1:
            return torch.where(x >= 0, x, self.weight * x)
        else:
            # Broadcast weight over the second dimension (channel dimension)
            # x is expected to have shape (N, C, ...) where C == num_parameters
            shape = [1, self.num_parameters] + [1] * (x.dim() - 2)
            return torch.where(x >= 0, x, self.weight.view(*shape) * x)


class CustomConstrainedLoss(nn.Module):
    def __init__(self, model, det_weight=1.0, margin=1e-4):
        """
        :param model: 传入你的 MLP 模型实例
        :param det_weight: 行列式惩罚项的权重系数，越大则约束越硬
        :param margin: 期望行列式至少大于这个微小的正值
        """
        super(CustomConstrainedLoss, self).__init__()
        self.model = model
        self.det_weight = det_weight
        self.margin = margin
        self.mse_fn = nn.MSELoss()
    def forward(self, output, target):
        # 1. 计算基本的 MSE 误差
        mse_loss = self.mse_fn(output, target)
        
        # 2. 计算行列式惩罚项
        det_penalty = 0.0
        # 遍历 net.net1 中的所有子模块
        for name, layer in self.model.net1.named_children():
            if isinstance(layer, nn.Linear):
                W = layer.weight
                # 检查是否为方阵 (rows == cols)
                if W.shape[0] == W.shape[1]:
                    # 计算行列式
                    d = torch.linalg.det(W)
                    # 惩罚项：如果 d < margin，则产生惩罚
                    # 使用 ReLU(margin - d) 确保当 d > margin 时导数为 0
                    det_penalty += torch.relu(self.margin - d)
                else:
                    # 如果不是方阵，行列式无定义
                    # 可选择跳过，或者针对长方形矩阵约束其奇异值的乘积（广义行列式）
                    pass
        
        # 总损失 = MSE + lambda * Penalty
        total_loss = mse_loss + self.det_weight * det_penalty
        return total_loss
# 使用示例：
# net = MLP(width=2, depth=2, w_in=2) # 假设 w_in=width 以确保方阵
# criterion = CustomConstrainedLoss(net, det_weight=0.1)
# loss = criterion(net(PUT), TARGET)