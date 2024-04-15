import os
import json
import pickle as pkl
import numpy as np
import torch
import utils
from tqdm import tqdm
import config
import random

def load_pkl(path):
    data=pkl.load(open(path,'rb'))
    return data
    
def read_hdf5(path):
    data=h5py.File(path,'rb')
    return data

def read_csv(path):
    data=pd.read_csv(path)
    return data

def read_csv_sep(path):
    data=pd.read_csv(path,sep='\t')
    return data
    
def dump_pkl(path,info):
    pkl.dump(info,open(path,'wb'))  
    
def read_json(path):
    utils.assert_exits(path)
    data=json.load(open(path,'rb'))
    '''in anet-qa returns a list'''
    return data

def pd_pkl(path):
    data=pd.read_pickle(path)
    return data

def read_jsonl(path):
    total_info=[]
    with open(path,'rb')as f:
        d=f.readlines()
    for i,info in enumerate(d):
        data=json.loads(info)
        total_info.append(data)
    return total_info

class Multimodal_Data():
    #mem, off, harm
    def __init__(self,opt,tokenizer,dataset,mode='train',few_shot_index=0):
        super(Multimodal_Data,self).__init__()
        self.opt=opt
        self.tokenizer = tokenizer
        self.mode=mode
        # opt.FEW_SHOT 为FALSE
        if self.opt.FEW_SHOT:
            self.few_shot_index=str(few_shot_index)
            self.num_shots=self.opt.NUM_SHOTS
            print ('Few shot learning setting for Iteration:',self.few_shot_index)
            print ('Number of shots:',self.num_shots)
        # 只有两类标签 opt.NUM_LABELS == 2
        self.num_ans=self.opt.NUM_LABELS
        #maximum length for a single sentence
        self.length=self.opt.LENGTH
        #maximum length of the concatenation of sentences
        self.total_length=self.opt.TOTAL_LENGTH
        self.num_sample=self.opt.NUM_SAMPLE
        # 是否添加实体信息
        # NOTE: 在添加完实体信息和人口统计信息后,训练结果更好
        self.add_ent=self.opt.ADD_ENT
        # 是否添加人口统计信息
        self.add_dem=self.opt.ADD_DEM
        print ('Adding exntity information?',self.add_ent)
        print ('Adding demographic information?',self.add_dem)
        self.fine_grind=self.opt.FINE_GRIND
        print ('Using target information?',self.fine_grind)
        # opt.FINE_GRIND == FALSE
        if opt.FINE_GRIND:
            #target information
            if self.opt.DATASET=='mem':
                self.label_mapping_word={0:'nobody',
                                         1:'race',
                                         2:'disability',
                                         3:'nationality',
                                         4:'sex',
                                         5:'religion'}
            elif self.opt.DATASET=='harm':
                self.label_mapping_word={0:'nobody',
                                         1:'society',
                                         2:'individual',
                                         3:'community',
                                         4:'organization'}
                self.attack_list={'society':0,
                                  'individual':1,
                                  'community':2,
                                  'organization':3}
                self.attack_file=load_pkl(os.path.join(self.opt.DATA,
                                                       'domain_splits','harm_trgt.pkl'))
            self.template="*<s>**sent_0*.*_It_was_targeting*label_**</s>*"
        else:
            self.label_mapping_word={0:self.opt.POS_WORD,
                                     1:self.opt.NEG_WORD}
            # 构建模板,这个 template 后面没有用到,只运行时打印出来
            self.template="*<s>**sent_0*.*_It_was*label_**</s>*"
            
        self.label_mapping_id={}
        # 将标签词(good bad) 和 vocab 中的 index 对应起来,存储在字典中
        for label in self.label_mapping_word.keys():
            mapping_word=self.label_mapping_word[label]
            #add space already
            assert len(tokenizer.tokenize(' ' + self.label_mapping_word[label])) == 1
            self.label_mapping_id[label] = \
            tokenizer._convert_token_to_id(
                tokenizer.tokenize(' ' + self.label_mapping_word[label])[0])
            # 将提示词和 tokenizer 中的 vocab 下标对应
            print ('Mapping for label %d, word %s, index %d' % 
                   (label,mapping_word,self.label_mapping_id[label]))
        #implementation for one template now
        # 输出信息
        self.template_list=self.template.split('*')
        print('Template:', self.template)
        print('Template list:',self.template_list)
        # 特殊字符字典
        self.special_token_mapping = {
            '<s>': tokenizer.convert_tokens_to_ids('<s>'),
            '<mask>': tokenizer.mask_token_id, 
            '<pad>': tokenizer.pad_token_id, #1 for roberta
            '</s>': tokenizer.convert_tokens_to_ids('<\s>') 
        }
        
        # self.opt.DEM_SAMP == Flase
        if self.opt.DEM_SAMP:
            print ('Using demonstration sampling strategy...')
            self.img_rate=self.opt.IMG_RATE
            self.text_rate=self.opt.TEXT_RATE
            self.samp_rate=self.opt.SIM_RATE
            print ('Image rage for measuring CLIP similarity:',self.img_rate)
            print ('Text rage for measuring CLIP similarity:',self.text_rate)
            print ('Sampling from top:',self.samp_rate*100.0,'examples')
            self.clip_clean=self.opt.CLIP_CLEAN
            clip_path=os.path.join(
                self.opt.CAPTION_PATH,
                dataset,dataset+'_sim_scores.pkl')
            print ('Clip feature path:',clip_path)
            self.clip_feature=load_pkl(clip_path)
        # support_examples 加载的 train 数据集中 entries 数据
        self.support_examples=self.load_entries('train')
        print ('Length of supporting example:',len(self.support_examples))
        # mode: train ? 重新加载了一遍？
        self.entries=self.load_entries(mode)
        # DEBUG: False
        if self.opt.DEBUG:
            self.entries=self.entries[:128]
        # 为后面构建 正负demostration 构建候选的 index 数组
        self.prepare_exp()
        print ('The length of the dataset for:',mode,'is:',len(self.entries))

    # 加载数据集
    def load_entries(self,mode):
        #print ('Loading data from:',self.dataset)
        #only in training mode, in few-shot setting the loading will be different
        if self.opt.FEW_SHOT and mode=='train':
            path=os.path.join(self.opt.DATA,
                              'domain_splits',
                              self.opt.DATASET+'_'+str(self.num_shots)+'_'+self.few_shot_index+'.json')
        else:            
            path=os.path.join(self.opt.DATA,
                              'domain_splits',
                              self.opt.DATASET+'_'+mode+'.json')
        # 加载数据集
        data=read_json(path)
        # 加载 caption 路径
        cap_path=os.path.join(self.opt.CAPTION_PATH,
                              self.opt.DATASET+'_'+self.opt.PRETRAIN_DATA,
                              self.opt.IMG_VERSION+'_captions.pkl')
        # 加载字幕
        captions=load_pkl(cap_path)
        entries=[]
        for k,row in enumerate(data):
            label=row['label']
            img=row['img']
            # mapping 模因对应的字幕
            cap=captions[img.split('.')[0]][:-1]#remove the punctuation in the end
            sent=row['clean_sent']
            #remember the punctuations at the end of each sentence
            # 此处的 caption 是模因字幕和图片中的文本结合在一起的
            cap=cap+' . '+sent+' . '
            #whether using external knowledge
            # self.add_ent 和 self.add_dem 都为 true
            if self.add_ent:
                cap=cap+' . '+row['entity']+' . '
            if self.add_dem:
                cap=cap+' . '+row['race']+' . '
            entry={
                'cap':cap.strip(),
                'label':label,
                'img':img
            }
            # few-shot == False
            if self.fine_grind:
                if self.opt.DATASET=='mem':
                    if label==0:
                        #[1,0,0,0,0,0]
                        entry['attack']=[1]+row['attack']
                    else:
                        entry['attack']=[0]+row['attack']
                elif self.opt.DATASET=='harm':
                    if label==0:
                        #[1,0,0,0,0,0]
                        entry['attack']=[1,0,0,0,0]
                    else:
                        attack=[0,0,0,0,0]
                        attack_idx=self.attack_list[self.attack_file[img]]+1
                        attack[attack_idx]=1
                        entry['attack']=attack
            entries.append(entry)
        return entries
    
    # 将文本编码
    def enc(self,text):
        return self.tokenizer.encode(text, add_special_tokens=False)
    # 为后续构建正负 demostration 准备供选择的下标数组
    def prepare_exp(self):                               
        ###add sampling
        support_indices = list(range(len(self.support_examples)))
        self.example_idx = []
        # self.num_sample = 1
        for sample_idx in tqdm(range(self.num_sample)):
            for query_idx in range(len(self.entries)):
                if self.opt.DEM_SAMP:
                    #filter dissimilar demonstrations
                    candidates= [support_idx for support_idx in support_indices
                                 if support_idx != query_idx or self.mode != "train"]
                    sim_score=[]
                    count_each_label = {label: 0 for label in range(self.opt.NUM_LABELS)}
                    context_indices=[]
                    clip_info_que=self.clip_feature[self.entries[query_idx]['img']]
                    
                    #similarity computation
                    for support_idx in candidates:
                        img=self.support_examples[support_idx]['img']
                        #this cost a lot of computation
                        #unnormalized: the same scale -- 512 dimension
                        if self.clip_clean:
                            img_sim=clip_info_que['clean_img'][img]
                        else:
                            img_sim=clip_info_que['img'][img]
                        text_sim=clip_info_que['text'][img]
                        total_sim=self.img_rate*img_sim+self.text_rate*text_sim
                        sim_score.append((support_idx,total_sim))
                    sim_score.sort(key=lambda x: x[1],reverse=True)
                    
                    #top opt.SIM_RATE entities for each label
                    num_valid=int(len(sim_score)//self.opt.NUM_LABELS*self.samp_rate)
                    """
                    if self.opt.DEBUG:
                        print ('Valid for each class:',num_valid)
                    """
                    
                    for support_idx, score in sim_score:
                        cur_label=self.support_examples[support_idx]['label']
                        if count_each_label[cur_label]<num_valid:
                            count_each_label[cur_label]+=1
                            context_indices.append(support_idx)
                else: 
                    #exclude the current example during training
                    context_indices = [support_idx for support_idx in support_indices
                                       if support_idx != query_idx or self.mode != "train"]
                #available indexes for supporting examples
                self.example_idx.append((query_idx, context_indices, sample_idx))

    # 构建正负 demostration
    def select_context(self, context_examples):
        """
        Select demonstrations from provided examples.
        """
        num_labels=self.opt.NUM_LABELS
        max_demo_per_label = 1
        counts = {k: 0 for k in range(num_labels)}
        if num_labels == 1:
            # Regression
            counts = {'0': 0, '1': 0}
        selection = []
        """
        # Sampling strategy from LM-BFF
        if self.opt.DEBUG:
            print ('Number of context examples available:',len(context_examples))
        """
        order = np.random.permutation(len(context_examples))
        for i in order:
            label = context_examples[i]['label']
            if num_labels == 1:
                # Regression
                #No implementation currently
                label = '0' if\
                float(label) <= median_mapping[self.args.task_name] else '1'
            if counts[label] < max_demo_per_label:
                selection.append(context_examples[i])
                counts[label] += 1
            if sum(counts.values()) == len(counts) * max_demo_per_label:
                break
        
        assert len(selection) > 0
        return selection
    
    # 将 prompt template 加入到 demostration,并对文本进行编码
    def process_prompt(self, examples, 
                       first_sent_limit, other_sent_limit):
        if self.fine_grind:
            prompt_arch=' It was targeting '
        else:
            # 默认模板是 'It was <mask>'
            prompt_arch=' It was '
        #currently, first and other limit are the same
        # 最后是三个 demostration 构成的文本 tokens
        input_ids = []
        attention_mask = []   
        # <mask> 的位置
        mask_pos = None # Position of the mask token
        # coancat_sent: 三个 demostration 拼接在一起
        concat_sent=""
        # examples[0]: 待推测的 demostration, examples[1\0]: 正负demostration
        for segment_id, ent in enumerate(examples):
            #tokens for each example
            new_tokens=[]
            if segment_id==0:
                #implementation for the querying example
                new_tokens.append(self.special_token_mapping['<s>'])
                # toekn 长度设置
                length=first_sent_limit
                # 待推测的 demostration 的 temp: 'It was <mask>. </s>'
                temp=prompt_arch+'<mask>'+' . </s>'
            else:
                length=other_sent_limit
                # self.fine_grind == False
                if self.fine_grind:
                    if ent['label']==0:
                        label_word=self.label_mapping_word[0]
                    else:
                        attack_types=[i for i, x in enumerate(ent['attack']) if x==1]
                        #only for meme
                        if len(attack_types)==0:
                            attack_idx=random.randint(1,5)
                        #randomly pick one
                        #already padding nobody to the head of the list
                        else:
                            order=np.random.permutation(len(attack_types))
                            attack_idx=attack_types[order[0]]
                        label_word=self.label_mapping_word[attack_idx]
                else:
                    label_word=self.label_mapping_word[ent['label']]
                # 正负 demostration 的 temp: 'It was good/bad. </s>'
                temp=prompt_arch+label_word+' . </s>'
            # 对 caption + sentence 进行编码
            new_tokens+=self.enc(' '+ent['cap'])
            # 构建好示例 prompt(正面 + 负面 + 待推理)
            #truncate the sentence if too long
            new_tokens=new_tokens[:length]
            # 对 template 进行编码
            new_tokens+=self.enc(temp)
            # 拼接 sentence + caption + template 构建一个demostration
            whole_sent=' '+ent['cap']+temp
            # 拼接 demostration
            concat_sent+=whole_sent
            #update the prompts
            input_ids+=new_tokens
            # 这些位置表示是非填充的
            attention_mask += [1 for i in range(len(new_tokens))]
        """
        if self.opt.DEBUG and self.opt.DEM_SAMP==False:
            print (concat_sent)
        """
        # 填充操作
        while len(input_ids) < self.total_length:
            input_ids.append(self.special_token_mapping['<pad>'])
            attention_mask.append(0)
        # 截取操作
        if len(input_ids) > self.total_length:
            input_ids = input_ids[:self.total_length]
            attention_mask = attention_mask[:self.total_length]
        mask_pos = [input_ids.index(self.special_token_mapping['<mask>'])]
        
        # Make sure that the masked position is inside the max_length
        assert mask_pos[0] < self.total_length
        result = {'input_ids': input_ids,
                  'sent':'<s>'+concat_sent,
                  'attention_mask': attention_mask,
                  'mask_pos': mask_pos}
        return result

                
    def __getitem__(self,index):
        #query item
        entry=self.entries[index]
        # 获取可以构建 正负 demostration 的样本下标
        #bootstrap_idx --> sample_idx
        query_idx, context_indices, bootstrap_idx = self.example_idx[index]
        #one example from each class
        # 构建完成正负 demostration
        supports = self.select_context(
            [self.support_examples[i] for i in context_indices])
        
        exps=[]
        exps.append(entry)
        exps.extend(supports)
        # exps[0]: 待推测的样本， exps[1\2]: 正负 demostration 
        prompt_features = self.process_prompt(
            exps,
            self.length,
            self.length
        )
            
        vid=entry['img']
        #label=torch.tensor(self.label_mapping_id[entry['label']])
        label=torch.tensor(entry['label'])
        target=torch.from_numpy(np.zeros((self.num_ans),dtype=np.float32))
        # 标签数组，用作 trian.py 的损失和评估模型
        # target = [0, 1] 
        target[label]=1.0

        cap_tokens=torch.Tensor(prompt_features['input_ids'])
        mask_pos=torch.LongTensor(prompt_features['mask_pos'])
        mask=torch.Tensor(prompt_features['attention_mask'])
        batch={
            'sent':prompt_features['sent'],
            'mask':mask,
            'img':vid,
            'target':target,
            'cap_tokens':cap_tokens,
            'mask_pos':mask_pos,
            'label':label
        }
        if self.fine_grind:
            batch['attack']=torch.Tensor(entry['attack'])
        #print (batch)
        return batch
        
    def __len__(self):
        return len(self.entries)
    
