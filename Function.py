import torch
import sympy
from TopoDataSyn import version_register

class version_info(version_register):
    def __init__(self):
        super().__init__(timeversion='260424-23:16')

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


class WeightConstraint_MLP_Loss(torch.nn.Module):
    def __init__(self, model, width=3, det_weight:float=100.0, margin:float=1e-4):
        """
        model: 传入你的 MLP 模型实例
        det_weight: 行列式惩罚项的权重系数，越大则约束越硬
        margin: 期望行列式至少大于这个微小的正值
        # 使用示例：
        # net = MLP(width=2, depth=2, w_in=2) # 假设 w_in=width 以确保方阵
        # criterion = CustomConstrainedLoss(net, det_weight=0.1)
        # loss = criterion(net(PUT), TARGET)
        """
        super(WeightConstraint_MLP_Loss, self).__init__()
        self.model = model
        self.width = width
        self.det_weight = det_weight
        self.margin = margin
        self.mse_fn = torch.nn.MSELoss()
        assert self.width <= self.model.net1[0].weight.shape[0] and self.width <= self.model.net1[0].weight.shape[1]
        assert self.width <= self.model.net2[0].weight.shape[0] and self.width <= self.model.net2[0].weight.shape[1]
    def forward(self, output, target):
        # 1. 计算基本的 MSE 误差
        mse_loss = self.mse_fn(output, target)
        #for name, param in self.model.named_parameters():
        #    if "weight" in name:
        #        print(f"{name}: {param.data}")
        # 2. 计算行列式惩罚项
        det_penalty = 0.0
        # 遍历 net.net1 中的所有子模块
        for name, layer in self.model.net1.named_children():
            if isinstance(layer, torch.nn.Linear):
                W = layer.weight[0:self.width, 0:self.width]
                d = torch.linalg.det(W)
                # 惩罚项：如果 d < margin，则产生惩罚
                # 使用 ReLU(margin - d) 确保当 d > margin 时导数为 0
                det_penalty += torch.relu(self.margin - d)

        for name, layer in self.model.net2.named_children():
            if isinstance(layer, torch.nn.Linear):
                W = layer.weight[0:self.width, 0:self.width]
                d = torch.linalg.det(W)
                # 惩罚项：如果 d < margin，则产生惩罚
                # 使用 ReLU(margin - d) 确保当 d > margin 时导数为 0
                det_penalty += torch.relu(self.margin - d)
        
        # 总损失 = MSE + lambda * Penalty
        total_loss = mse_loss + self.det_weight * det_penalty
        return total_loss


def analyze_tensor(matrix_tensor: torch.Tensor):
    """
    对于一个torch.tensor:
    - 如果是方阵，则依次返回Jordan分解、行列式、奇异值分解；
    - 如果不是方阵，则返回False，False，奇异值分解。
    """
    if matrix_tensor.dim() != 2:
        raise ValueError("输入的 tensor 必须是二维矩阵")
    # 奇异值分解 SVD (对所有矩阵都适用)
    # torch.linalg.svd 返回 U, S, Vh (其中 matrix = U @ diag(S) @ Vh)
    # 注意：根据 PyTorch 版本，默认可能返回 (U, S, Vh)
    svd_result = torch.linalg.svd(matrix_tensor)
    # 判断是否为方阵
    is_square = matrix_tensor.shape[0] == matrix_tensor.shape[1]
    if is_square:
        # 1. 计算行列式
        # 注意: torch.linalg.det 要求输入 tensor 的数据类型为浮点型或复数型
        det_result = torch.linalg.det(matrix_tensor.to(torch.float32))
        # 2. 计算 Jordan 分解 (借助 SymPy)
        try:
            # 将 PyTorch Tensor 转换为 numpy 数组，再转换为 SymPy 矩阵
            sympy_matrix = sympy.Matrix(matrix_tensor.detach().cpu().numpy())
            # jordan_form() 返回 (P, J)，使得 original_matrix = P * J * P**-1
            P, J = sympy_matrix.jordan_form()
            jordan_result = (P, J)
        except Exception as e:
            print(f"Jordan 分解计算失败: {e}")
            jordan_result = None
        return jordan_result, det_result, svd_result
    else:
        return False, False, svd_result
