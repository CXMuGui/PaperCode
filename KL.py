import torch
import torch.nn.functional as F
import torch.nn as nn

class KL_Loss(nn.Module):
    def __init__(self):
        super(KL_Loss, self).__init__()
    # 参数 两个不同的概率分布x y
    def forward(self, x, y):
        # 使用F.normalize确保两个概率分布的和为1
        x_distribution = F.normalize(x, p=1, dim=-1)
        y_distribution = F.normalize(y, p=1, dim=-1)
        # x_distribution = F.softmax(x, dim=-1)
        # y_distribution = F.softmax(y, dim=-1)

        # 定义损失函数进行计算
        KL_loss = F.kl_div(x_distribution.log(), y_distribution, reduction='batchmean')
        # 如果有多个样本（批次）取平均值
        return KL_loss

if __name__ == "__main__":
    loss = KL_Loss()

    # 输入两个概率分布
    x_distribution = torch.tensor([0.4, 0.6])
    y_distribution = torch.tensor([0.3, 0.7])

    # 进行计算
    loss_value = loss.forward(x_distribution, y_distribution)

    print(f"KL Divergence Loss: {loss_value.item()}")
