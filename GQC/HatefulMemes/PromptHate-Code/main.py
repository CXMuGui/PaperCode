import torch
import numpy as np
import random

import config
import os
from train import train_for_epoch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer

# 设置随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

if __name__=='__main__':
    # 导入配置参数文件
    opt=config.parse_opt()
    torch.cuda.set_device(opt.CUDA_DEVICE)
    set_seed(opt.SEED)
    
    # Create tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')

    constructor='build_baseline'
    # opt.MODEL 值为 pbm
    if opt.MODEL=='pbm':
        # 从 dataset.py 文件中导入数据预处理类，作用是将从本地加载的文本处理为带有提示模板的模型文本输入
        from dataset import Multimodal_Data
        # 模型文件
        import baseline
        # 处理训练集，经过 Multimodal_Data 处理后的单个样本形式为：
        # 由 3 个 demostrations(正类、负类、待预测)组成，其中一个 demostration 由 meme caption + meme text + prompt template
        train_set=Multimodal_Data(opt,tokenizer,opt.DATASET,'train',opt.SEED-1111)
        test_set=Multimodal_Data(opt,tokenizer,opt.DATASET,'test')
        # label_mapping_id: {0: 205, 1: 1099}
        # label_mapping_wrod: {0: 'good', 1: 'bad'}
        # label_list: [205, 1099]
        label_list=[train_set.label_mapping_id[i] for i in train_set.label_mapping_word.keys()]
        model=getattr(baseline,constructor)(opt, label_list).cuda()
    else:
        from roberta_dataset import Roberta_Data
        import roberta_baseline
        train_set=Roberta_Data(opt,tokenizer,opt.DATASET,'train',opt.SEED-1111)
        test_set=Roberta_Data(opt,tokenizer,opt.DATASET,'test')
        model=getattr(roberta_baseline,constructor)(opt).cuda()
        
    train_loader=DataLoader(train_set,
                            opt.BATCH_SIZE,
                            shuffle=True,
                            num_workers=1)
    test_loader=DataLoader(test_set,
                           opt.BATCH_SIZE,
                           shuffle=False,
                           num_workers=1)
    train_for_epoch(opt,model,train_loader,test_loader)
    
    exit(0)
    