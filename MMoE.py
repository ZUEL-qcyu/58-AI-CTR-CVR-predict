import torch
import torch.nn as nn
from DCN import DCNExpert

class MMoE(nn.Module):
    """
    Multi-gate Mixture-of-Experts (MMoE) 模型，使用DCN作为专家网络
    """
    def __init__(self, input_dim,
                 num_experts=3,
                 num_tasks=2,
                 expert_dim=64,
                 expert_hidden_dims=[128, 64],
                 gate_hidden_dims=[64],
                 task_hidden_dims=[32],
                 cross_layers=2):

        super(MMoE, self).__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.expert_dim = expert_dim

        # 专家网络 - 使用DCN
        self.experts = nn.ModuleList([
            DCNExpert(input_dim,expert_dim, cross_layers, expert_hidden_dims)
            for _ in range(num_experts)
        ])

        # 每个任务的门控网络
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, gate_hidden_dims[0]),
                nn.BatchNorm1d(gate_hidden_dims[0]),
                nn.ReLU(),
                nn.Linear(gate_hidden_dims[0], num_experts),
                nn.Softmax(dim=1)
            ) for _ in range(num_tasks)
        ])

        # 每个任务的塔网络(Tower Networks)
        self.towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(expert_dim, task_hidden_dims[0]),
                nn.BatchNorm1d(task_hidden_dims[0]),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(task_hidden_dims[0], 1),
                nn.Sigmoid()
            ) for _ in range(num_tasks)
        ])

    def forward(self, x):
        # 获取所有专家的输出
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))
        expert_outputs = torch.stack(expert_outputs, dim=1)  # (batch_size, num_experts, expert_dim)

        # 计算每个任务的门控权重并应用到专家输出
        task_outputs = []
        for task_id in range(self.num_tasks):
            gate_output = self.gates[task_id](x).unsqueeze(2)  # (batch_size, num_experts, 1)
            # 加权组合所有专家的输出
            combined_experts = torch.sum(expert_outputs * gate_output, dim=1)  # (batch_size, expert_dim)
            # 通过塔网络得到任务特定的预测
            task_output = self.towers[task_id](combined_experts)
            task_outputs.append(task_output)

        return task_outputs  # 返回每个任务的预测结果