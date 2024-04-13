## 我这边的环境版本介绍：

python == 3.8
pytorch==1.12.1 
torchvision==0.13.1 
torchaudio==0.12.1 
cudatoolkit=11.3

scikit-learn==1.3.2
transformers==4.39.9

## NOTE：第一次运行的时候，需要联网下载 Roberta-large 模型，如果服务器隔离外网可能需要进行反向代理来下载 Roberta 模型（我是采取这种方式）



## 运行：

- 切换自己的路径： cd your/filepath/prompthate-main/PromptHate-Code/

- 设置路径，在 config.py 文件里面设置
    parser.add_argument('--DATA',type=str,default='/212023085404022/workspace/PaperCode/HatefulMemes/data')
    parser.add_argument('--CAPTION_PATH',type=str,default='/212023085404022/workspace/PaperCode/HatefulMemes/caption')
    
- 两种方式运行：
  1.python main.py
    以默认的参数运行，只对 mem 这个数据集进行训练
  
  2.bash run.sh
    自己设置的 10 个随机种子和学习率，分别对 mem 和 harm 两个数据集进行训练
  
  

## 重要文件介绍：

**config.py** : 配置参数，batch_size 默认是 16，CUDA_DEVICE 默认是 1

**main.py** : 主函数

**dataset.py** : 读取预处理好的数据集，主要内容是 " it is <mask> " prompt 模板构建，和 demostration 构建。一个 demostration = Meme Text + Image Caption + Template。最后返回的是下面三个拼接的 demostration。
![image](https://github.com/CXMuGui/PaperCode/assets/86507078/773440e0-604e-469f-829e-448e852a5df6)

**baseline.py** : 模型文件，预测 <mask> 这个位置上所有单词（vocab）的得分，取得自己指定的提示词在 <mask> 上的得分（这里提示词是 good 和 bad），最后返回到 trian.py 算损失。

**train.py** : 模型训练文件

**run.sh** : 脚本运行文件，分别运行 mem 和 harm 这两个数据集。且指定了 10 个随机种子和学习率。
