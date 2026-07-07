import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn

class Expert(nn.Module):
    """专家网络：简单的全连接网络"""
    def __init__(self, input_dim):
        super(Expert, self).__init__()
        self.expert = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, input_dim)
        )

    def forward(self, x):
        x = self.expert(x)
        return x

class GatingNetwork(nn.Module):
    """门控网络：选择Top-K个专家并重新计算权重，同时返回选择的专家索引"""
    def __init__(self, input_dim, num_experts, topk=1):
        super(GatingNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, num_experts)
        self.softmax = nn.Softmax(dim=-1)
        self.topk = topk  # 选择Top-K个专家

    def forward(self, x):
        # x shape: [B, input_dim]
        
        # 1. 计算原始专家权重（未归一化）
        raw_scores = self.fc(x)  # [B, num_experts]
        
        # 2. 选择Top-K个专家的下标
        # values: [B, topk]  topk的原始分数
        # indices: [B, topk] 对应的专家索引
        topk_values, topk_indices = torch.topk(raw_scores, k=self.topk, dim=-1)
        
        # 3. 创建mask，只保留Top-K专家的权重
        # 初始化全为负无穷（softmax后接近0）
        mask = torch.full_like(raw_scores, float('-inf'))
        # 将Top-K专家的位置设为原始分数
        mask.scatter_(dim=-1, index=topk_indices, src=topk_values)
        
        # 4. 对mask后的分数重新计算softmax
        gated_weights = self.softmax(mask)  # [B, num_experts]，非Top-K位置为0
        
        # 返回权重和选择的专家索引
        return gated_weights, topk_indices, raw_scores    #raw_scores用于视角专家分配

class MoVE(nn.Module):
    """混合专家模型：组合多个专家和一个门控网络，返回每个专家对应的样本索引"""
    def __init__(self, input_dim, view_experts=3):
        super(MoVE, self).__init__()
        self.num_experts = view_experts
        self.topk = 1  # 从门控网络获取topk值，这里保持一致
        
        self.shared_experts = Expert(input_dim)
        # 创建多个专家网络
        self.view_experts = nn.ModuleList([
            Expert(input_dim) 
            for _ in range(view_experts)
        ])
        
        # 创建门控网络
        self.gate = GatingNetwork(input_dim, view_experts, topk=self.topk)
        
    def forward(self, x):
        # x shape: (batch_size, input_dim)
        batch_size = x.size(0)
        
        # 获取每个专家的权重和选择的专家索引
        # gate_weights: (batch_size, num_experts)
        # expert_indices: (batch_size, topk) - 每个样本选择的专家索引
        gate_weights, expert_indices, raw_scores = self.gate(x)
        
        # 每个专家的输出 (batch_size, output_dim)
        expert_outputs = [expert(x) for expert in self.view_experts]
        
        # 将专家输出堆叠 (num_experts, batch_size, output_dim)
        expert_outputs = torch.stack(expert_outputs)
        
        # 转置为 (batch_size, num_experts, output_dim)
        expert_outputs = expert_outputs.permute(1, 0, 2)
        
        # 门控权重与专家输出相乘并求和 (batch_size, output_dim)
        # 扩展门控权重维度以便广播 (batch_size, num_experts, 1)
        gate_weights = gate_weights.unsqueeze(-1)
        moe_output = torch.sum(gate_weights * expert_outputs, dim=1)

        moe_output += self.shared_experts(x)
        
        # --------------------------
        # 生成每个专家对应的样本索引mask
        # --------------------------
        expert_sample_masks = []
        # 遍历每个专家
        for expert_id in range(self.num_experts):
            # 创建mask，初始化为False
            mask = torch.zeros(batch_size, dtype=torch.bool, device=x.device)
            # 检查每个样本是否选择了当前专家
            # expert_indices形状: [batch_size, topk]
            for sample_idx in range(batch_size):
                # 检查当前样本的topk专家中是否包含当前专家ID
                if expert_id in expert_indices[sample_idx]:
                    mask[sample_idx] = True
            # 记录专家ID和对应的样本mask
            expert_sample_masks.append(mask)
        
        # 返回模型输出和每个专家对应的样本mask列表
        return moe_output, expert_sample_masks, raw_scores

# 使用示例
if __name__ == "__main__":
    # 配置参数
    input_dim = 64
    num_experts = 3
    batch_size = 5
    topk = 1  # 每个样本选择1个专家
    
    # 创建模型
    model = MoVE(input_dim, view_experts=num_experts)
    
    # 创建随机输入
    x = torch.randn(batch_size, input_dim)
    
    # 前向传播
    output, expert_sample_masks, raw_scores = model(x)
    
    # 打印结果
    print(raw_scores)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print("\n每个专家对应的样本索引:")
    for mask in expert_sample_masks:
        # 获取被当前专家处理的样本索引
        print(mask)
        sample_indices = torch.where(mask)[0].tolist()
        print(f"处理的样本索引: {sample_indices}")
    
    
