import torch
import torch.nn as nn
from transformers import RobertaForMaskedLM
from transformers import RobertaTokenizer, RobertaModel

class RobertaPromptModel(nn.Module):
    def __init__(self,label_list):
        super(RobertaPromptModel, self).__init__()
        # label_list: [205, 1099]
        self.label_word_list=label_list
        self.roberta = RobertaForMaskedLM.from_pretrained('roberta-large')

    def forward(self,tokens,attention_mask,mask_pos,feat=None):
        batch_size = tokens.size(0)
        #the position of word for prediction
        if mask_pos is not None:
            mask_pos = mask_pos.squeeze()
        
        # out: logits 是模型对于每一个 token 在词汇表中每个可能此的预测分数
        # shape: [batch_size, sequence_length, vocab_size]
        out = self.roberta(tokens, 
                           attention_mask)
        # predict_mask_scores: 每一句文本在 mask_pos 的tokens 的预测值（一个位置有 vocab_size 多的预测值）
        # shape: [batch_size,vocab_size]
        prediction_mask_scores = out.logits[torch.arange(batch_size),
                                          mask_pos]
        
        logits = []
        # 遍历两遍，分别取出 good 和 bad 这两个 token 位置的预测概率
        for label_id in range(len(self.label_word_list)):
            logits.append(prediction_mask_scores[:,
                                                 self.label_word_list[label_id]
                                                ].unsqueeze(-1))
        logits = torch.cat(logits, -1)
        # loigts: shape = [batch_size, 2]
        return logits
        
    
def build_baseline(opt,label_list):  
    print (label_list)
    return RobertaPromptModel(label_list)

    