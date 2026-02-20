import torch

# 自定义带可学习参数的激活函数
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
        if self.num_parameters == 1:
            return torch.where(x >= 0, x, self.weight * (torch.exp(x) - 1))
        else:
            # Broadcast weight over the second dimension (channel dimension)
            # x is expected to have shape (N, C, ...) where C == num_parameters
            shape = [1, self.num_parameters] + [1] * (x.dim() - 2)
            return torch.where(x >= 0, x, self.weight.view(*shape) * (torch.exp(x) - 1))