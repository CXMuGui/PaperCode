import torch
import torch.nn as nn
import ot


# 封装OT损失函数
class OTLoss(nn.Module):
    def __init__(self):
        super(OTLoss, self).__init__()

    def forward(self, source_distribution, target_distribution, source_samples, target_samples):
        # 计算样本之间的距离矩阵
        distance_matrix = torch.norm(source_samples[:, None] - target_samples[None, :], dim=-1)

        # 计算概率分布中的每个元素除以总和，确保概率分布的和为一（进行归一化）
        # 复制概率分布 防止对需要梯度的张量进行原地操作
        source_distribution = source_distribution.clone().detach()
        target_distribution = target_distribution.clone().detach()

        source_distribution /= source_distribution.sum()
        target_distribution /= target_distribution.sum()

        # 将PyTorch张量转换为NumPy数组
        source_distribution_np = source_distribution.detach().cpu().numpy()
        target_distribution_np = target_distribution.detach().cpu().numpy()

        # 使用emd2函数计算了EMD即Wasserstein距离   ot_loss表示将源概率分布转移到目标概率分布的所需最小工作量
        ot_loss = ot.emd2(source_distribution_np, target_distribution_np, distance_matrix.cpu().numpy())

        # 将NumPy数组转换回PyTorch张量
        ot_loss = torch.tensor(ot_loss, dtype=torch.float32, requires_grad=True)

        return ot_loss


if __name__ == "__main__":
    ot_loss = OTLoss()

    # 输入样本分布
    source_distribution = torch.tensor([0.4, 0.6], requires_grad=True)
    target_distribution = torch.tensor([0.3, 0.7], requires_grad=True)
    source_samples = torch.tensor([[0.07], [0.04]])
    target_samples = torch.tensor([[0.001], [0.02]])

    # 进行计算
    loss_value = ot_loss(source_distribution, target_distribution, source_samples, target_samples)

    print(f"OT Loss using Euclidean Distance: {loss_value.item()}")
