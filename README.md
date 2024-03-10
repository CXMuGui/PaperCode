## OT损失函数

导入库

```Python
import torch
import torch.nn as nn
import ot
```

函数所需参数为

```python
source_distribution: 源概率分布，一个表示源样本的概率分布的张量。

target_distribution: 目标概率分布，一个表示目标样本的概率分布的张量。

source_samples: 源样本，一个表示从源概率分布中采样得到的样本的张量。

target_samples: 目标样本，一个表示从目标概率分布中采样得到的样本的张量。
```



使用emd2函数进行计算最小工作量

```python
# 使用emd2函数计算了EMD即Wasserstein距离  ot_loss表示将源概率分布转移到目标概率分布的所需最小工作量
ot_loss = ot.emd2(source_distribution_np, target_distribution_np, distance_matrix.cpu().numpy())
```



## KL散度损失函数

导入库

```python
import torch
import torch.nn.functional as F
import torch.nn as nn
```

函数所需输入参数为

```python
两个参数 x 和 y，它们都是表示概率分布的张量
x: 一个概率分布的张量，用于计算KL散度损失。
y: 另一个概率分布的张量，与 x 对比计算KL散度损失。
```



在函数内部，它使用 `F.normalize` 确保两个概率分布的和为1。然后，通过计算KL散度损失（KL散度的定义涉及对数运算），并且通过设置 `reduction='batchmean'` 进行批次平均。最后，返回计算得到的KL散度损失。

