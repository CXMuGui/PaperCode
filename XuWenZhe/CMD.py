import torch.nn as nn
import torch


class CMD(nn.Module):

    def __init__(self):
        super(CMD, self).__init__()

    def forward(self, x1, x2, n_moments): # 在该代码中，E(x)只取了第零位的均值
        
        # 获取均值E(x1)和E(x2)
        mx1 = torch.mean(x1, 0)
        mx2 = torch.mean(x2, 0)
        print("E(X1):")
        print(mx1)

        print("\nE(X2):")
        print(mx2)
        # 获取 H-E(X)
        sx1 = x1-mx1
        sx2 = x2-mx2
        # 获取||E(x1)+E(x2)||2
        dm = self.matchnorm(mx1, mx2)
        print("\n||E(x1)+E(x2)||2:")
        print(dm)
        scms = dm
        for i in range(n_moments - 1):
            print("H-E(X1):")
            print(sx1)
            print("\nH-E(X2)():")
            print(sx2)
            scms += self.scm(sx1, sx2, i + 2)
            print("\n||Ck1()+Ck2()||2:")
            print(sx2)
        return scms

    def matchnorm(self, x1, x2): #计算两数之差的二范式
        power = torch.pow(x1-x2,2)
        summed = torch.sum(power)
        sqrt = summed**(0.5)
        return sqrt
        # return ((x1-x2)**2).sum().sqrt()

    def scm(self, sx1, sx2, k):
        ss1 = torch.mean(torch.pow(sx1, k), 0)
        ss2 = torch.mean(torch.pow(sx2, k), 0)
        print("Ck1():")
        print(ss1)

        print("\nCk2():")
        print(ss2)
        return self.matchnorm(ss1, ss2)

if __name__ == "__main__":
    # cmd测试
    cmd = CMD()
    # 生成两个随机维度为 (a, b) 和 (c, b) 维度的张量
    torch.manual_seed(42)
    random_tensor_1 = torch.randn((2, 4))
    random_tensor_2 = torch.randn((3, 4))

    # 打印生成的张量
    print("Random Tensor 1:")
    print(random_tensor_1)

    print("\nRandom Tensor 2:")
    print(random_tensor_2)
    loss = cmd(random_tensor_1,random_tensor_2,2)
    print(loss)
