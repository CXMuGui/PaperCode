
# 多模态特征融合模块，三种模态融合

# 融合模块，将添加了 prompt 的 sentece embedding 和 img 和 caption 特征进行融合
class MAG(nn.Module):
    def __init__(self, opt, args):
        super(MAG,self).__init__()
        # MAG 的对齐方法为 ctc 不是 sim
        self.alignNet = AlignSubNet(args, 'ctc')
        
        # 字幕特征和图像特征映射到文本特征上
        self.W_h_img = nn.Linear(args.img_feat_dim + args.text_feat_dim, args.text_feat_dim)
        self.W_h_cap = nn.Linear(args.cap_feat_dim + args.text_feat_dim, args.text_feat_dim)

        self.W_img = nn.Linear(args.img_feat_dim, args.text_feat_dim)
        self.W_cap = nn.Linear(args.cap_feat_dim, args.text_feat_dim)    
        
        self.LayerNorm = nn.LayerNorm(args.hidden_size)
        # 0.006
        self.beta_shift = args.beta_shift
        self.dropout = nn.Dropout(args.dropout_prob)

    def forward(self, text_embedding, img_feat, cap_feat):
        eps = 1e-6
        
        # 进行对齐
        aligned_text_embedding, aligned_img, aligned_cap = self.alignNet(text_embedding,img_feat,cap_feat)
        
        # 将字幕特征和图像特征映射到文本特征上
        weight_img = F.relu(self.W_h_img(torch.cat((aligned_img, aligned_text_embedding), dim=-1)))
        weight_cap = F.relu(self.W_h_cap(torch.cat((aligned_cap, aligned_text_embedding), dim=-1)))
        
        # 图像和caption特征的融合向量
        h_m = weight_img * self.W_img(aligned_img) + weight_cap * self.W_cap(aligned_cap)
        
        # 文本嵌入的 L2 范数
        em_norm = aligned_text_embedding.norm(2, dim=-1)
        # 是融合特征 h_m 的 L2 范数。
        hm_norm = h_m.norm(2, dim=-1)
        
        hm_norm_ones = torch.ones(hm_norm.shape, requires_grad=True).to(aligned_text_embedding.device)
        hm_norm = torch.where(hm_norm == 0, hm_norm_ones, hm_norm)
        
        thresh_hold = (em_norm / (hm_norm + eps)) * self.beta_shift
        
        ones = torch.ones(thresh_hold.shape, requires_grad=True).to(aligned_text_embedding.device)
        
        alpha = torch.min(thresh_hold, ones)
        alpha = alpha.unsqueeze(dim=-1)
        
        img_cap_embedding = alpha * h_m
        
        embedding_output = self.dropout(
            self.LayerNorm(img_cap_embedding + aligned_text_embedding)
        )
        
        return embedding_output