# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging
from os.path import join as pjoin
logger = logging.getLogger(__name__)
import torch
import torch.nn as nn
from torch.nn import functional as F
import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
import warnings
import math
from timm.layers import Mlp, DropPath
from torchvision.models import resnet18, ResNet18_Weights
from .modules import MoVE
import numpy as np
from einops import rearrange


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = 16
        ctx_init = ""
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution


        ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
        #ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        #prompt_prefix = " ".join(["X"] * n_ctx)

        self.ctx = nn.Parameter(ctx_vectors)  
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [name for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p, truncate=True) for p in prompts])
        #print(tokenized_prompts.shape, '?????')
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  
        #self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :]) 
        self.register_buffer("token_suffix", embedding[:, 1:, :]) 

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts 
        self.name_lens = name_lens
        self.class_token_position = "end"

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  
                    ctx,     
                    suffix, 
                ],
                dim=1,
            )
            prompts = prompts[:, :77, :]

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     
                        ctx_i_half1,  
                        class_i,      
                        ctx_i_half2,  
                        suffix_i,   
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  
                        class_i,   
                        ctx_i,    # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        else:
            raise ValueError
        return prompts


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

class Attention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        #self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        raw_attn = q @ k.transpose(-2, -1)
        attn = raw_attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, raw_attn
    
class Block(nn.Module):
    def __init__(self, dim, num_heads=12, mlp_ratio=4.,drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=True,
            attn_drop=0.1,
            proj_drop=0.1,
            norm_layer=norm_layer,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x, attn = self.attn(self.norm1(x))
        x = x + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn[:, :, 0, 1:].softmax(dim=-1)

class AttentionAggregation(nn.Module):
    """
    从ABMIL中提取的独立注意力聚合模块
    功能：输入Bag内的实例特征，输出注意力权重、聚合后的Bag特征
    完全保留原ABMIL的注意力计算逻辑
    """
    def __init__(self, feat_dim: int = 512, hidden_dim: int = 128, num_class: int = 5, pool: str = "softmax"):
        super().__init__()
        # 原ABMIL的注意力参数
        self.pool = pool
        self.num_class = num_class
        
        # 注意力分支a (Tanh) + 分支b (Sigmoid)
        self.attention_a = nn.Sequential(nn.Linear(feat_dim, hidden_dim), nn.Tanh())
        self.attention_b = nn.Sequential(nn.Linear(feat_dim, hidden_dim), nn.Sigmoid())
        self.attention_c = nn.Linear(hidden_dim, 1)  # 输出1维：每个实例的注意力权重

    def forward(self, ins_feat):
        """
        前向传播：输出单个Bag聚合特征
        :param ins_feat: 单个Bag的实例特征，shape [num_ins, feat_dim]
        :return:
            agg_bag_feat: 聚合后的Bag特征，shape [feat_dim]
            att_weight: 实例注意力权重，shape [num_ins]
        """
        # 1. 计算实例级注意力权重（原逻辑简化：输出1维权重）
        a = self.attention_a(ins_feat)  # [num_ins, hidden_dim]
        b = self.attention_b(ins_feat)  # [num_ins, hidden_dim]
        A = a.mul(b)                    # 逐元素相乘 [num_ins, hidden_dim]
        A = self.attention_c(A)         # [num_ins, 1] → 每个实例的注意力分数
        
        # 2. 注意力权重归一化（沿实例维度）
        A = A.squeeze(1)  # 去掉最后一维 → [num_ins]
        if self.pool == "sigmoid":
            A = F.sigmoid(A)
        else:
            A = F.softmax(A, dim=0)     # 所有实例权重和为1
        
        # 3. 加权聚合为单Bag特征（核心修正：权重@实例特征 → [feat_dim]）
        agg_bag_feat = torch.matmul(A.unsqueeze(0), ins_feat).squeeze(0)  # [1, feat_dim] → [feat_dim]
        
        return agg_bag_feat, A

class SimplePoolingAggregation(nn.Module):
    """
    极简的均值/最大值聚合模块
    功能：根据指定的pooling_method，对输入特征沿指定维度做mean/max聚合
    """
    def __init__(self, pooling_method: str = "mean", dim: int = 0):
        super().__init__()
        self.pooling_method = pooling_method  # 聚合方式：mean/max
        self.dim = dim                        # 聚合的维度（默认dim=0，适配你的场景）

    def forward(self, x):
        """
        前向传播：执行mean/max聚合
        :param x: 输入特征，shape [N, D]（比如N=实例数，D=特征维度）
        :return: 聚合后的特征，shape [D]
        """
        #print(x.shape, '.....')
        if 'mean' == self.pooling_method:
            x = x.mean(dim=self.dim)
        elif 'max' == self.pooling_method:
            x = torch.max(x, dim=self.dim)[0]  # [0]取max的数值，舍弃索引
            #print(x.shape,'??????')
        return x, None


class FAA_LARGE_tmed(nn.Module):
    def __init__(self, config, n_classes=5):
        super(FAA_LARGE_tmed, self).__init__()

        self.config = config
        self.dim = 512

        self.feature_extractor = resnet18(weights=ResNet18_Weights)
        self.feature_extractor.fc = nn.Sequential(nn.Linear(512, self.dim), nn.ReLU())

        #freeze in training
        clip_model, _ = clip.load("RN50", device="cpu")
        self.prompt_learner = PromptLearner(config['text_prompt'], clip_model.float())
        self.text_encoder = TextEncoder(clip_model.float())
        self.text_adapter = nn.Linear(1024, self.dim)

        

        self.a = nn.Parameter(torch.tensor(self.config['ratio']))

        
        # 多层 Transformer
        if config['aggr'] == 'attn':
            self.layer = AttentionAggregation(feat_dim=self.dim, hidden_dim=self.dim//4, num_class=n_classes)
        elif config['aggr'] == 'transformer':
            self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))
            self.layer1 = Block(dim=self.dim, num_heads=4)
            self.layer2 = Block(dim=self.dim, num_heads=4)
        elif config['aggr'] == 'simple':
            self.layer = SimplePoolingAggregation(pooling_method='max', dim=0)


        self.move = MoVE(self.dim, view_experts=config['views'])
        self.views = config['views']

       #self.MIL_f = MIL_f
        self.n_classes = n_classes

        self.norm = nn.LayerNorm(self.dim)
        self.fc = nn.Sequential(nn.Linear(self.dim, self.dim//4),
                                nn.ReLU(),
                                nn.Dropout(0.3),
                                nn.Linear(self.dim//4, n_classes))

        self.ce = nn.CrossEntropyLoss(torch.tensor([3.0, 2.0, 1.0]).cuda())
        self.semi_ce = nn.CrossEntropyLoss(torch.tensor([3.0, 2.0, 1.0]).cuda())
        self.view_ce = nn.CrossEntropyLoss(torch.tensor([3.0, 3.0, 1.0]).cuda())

    
    def MFS(self, ins_feat, prompts):

        #no choice
        if len(ins_feat) == 0:
            return ins_feat, None, None
        

        feats = F.normalize(ins_feat, p=2, dim=-1)  #shape [n d]
        pro = F.normalize(prompts, p=2, dim=-1)  #shape [c d]

        text_relevance = feats @ pro.t()    #shape [n c]

        relevance = F.softmax(text_relevance/0.01, dim=-1)

        relevance, _ = relevance.max(dim=-1)

        
        n, _ = text_relevance.shape
        if n == 1:
            threshold = 0
        else:
            threshold = relevance.mean() + self.a*relevance.std()
                #threshold = relevance.mean() + self.config['ratio']*relevance.std()

        selected_indices = relevance >= threshold
            
        selected_features = ins_feat[selected_indices]   # [n d]s

        if len(selected_features) == 0:
            topk_values, topk_indices = torch.topk(relevance, k=1, dim=-1)
            selected_features = ins_feat[topk_indices]
            selected_indices = torch.zeros_like(relevance, dtype=torch.bool)
            selected_indices[topk_indices] = True
        
        return selected_features, selected_indices, text_relevance
    
    
    def PPL(self, s_fi, t_r, target_class):
        
        if t_r is None:
            return 0
        # normal
        device = t_r.device       
    
        pred = t_r[s_fi]
        if len(pred) == 0:
            return 0

        n, _ = pred.shape

        t_c = torch.full((n,), target_class).to(device)
        loss = self.semi_ce(pred/0.07, t_c)

        return loss

    def AELoss(self, v_scores, view_class):
        return self.view_ce(v_scores, view_class)
    
    def forward(self, x, len_list, **kwargs):

        x = self.feature_extractor(x)

        prompts = self.prompt_learner()
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = self.text_adapter(text_features)

        ini_idx = 0

        y = []
        patch_loss = torch.tensor(0.).to(x.device)
        view_loss = torch.tensor(0.).to(x.device)

        labels = kwargs['labels']
        view_labels = kwargs['view_labels']

        for i, length in enumerate(len_list):
            ins_feat = x[ini_idx : ini_idx + length]
            ini_idx += length

            #move 视角分配
            moe_output, expert_sample_masks, raw_scores = self.move(ins_feat)
            view_labels[i] = view_labels[i].to(x.device)
            view_loss += self.AELoss(raw_scores, view_labels[i])

            h = []

            for i in range(self.views):
                s_f, s_fi, t_r = self.MFS(moe_output[expert_sample_masks[0]], text_features[i*(self.n_classes) : (i+1)*(self.n_classes)])
                h.append(s_f)
                #PPL
                patch_loss += self.PPL(s_fi, t_r, labels[i])
        

            #filter converge
            h = torch.concat(h, dim=0)

            B = h.shape[0]
            if self.config['aggr'] == 'transformer':
                cls_tokens = self.cls_token.expand(1, -1, -1)
                h = h.unsqueeze(0) 
                h = torch.cat((cls_tokens, h), dim=1)
                
            # Transformer 整体聚合
            h, attn = self.layer1(h)
            #h = self.norm(h)
            h, attn = self.layer2(h)

            h = self.norm(h)


            y.append(h[:, 0])

        h = torch.stack(y, dim=0)    #h [B dim]
        #h = rearrange(h, 'b c d -> (b c) d')
        if self.config['aggr'] == 'transformer':
            h = rearrange(h, 'b c d -> (b c) d')

        logits = self.fc(h) #[B, n_classes]

        loss = (patch_loss + view_loss) / len(len_list)
        cls_loss = self.ce(logits, labels) 

        loss = 1.0 * cls_loss + 0.3 * loss

        results_dict = {'predicted': logits, 
                        'loss': loss,
                        'features': h,   #t-SNE  可视化
                        }        
        return results_dict
