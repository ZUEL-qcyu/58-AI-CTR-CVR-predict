import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossLayer(nn.Module):
    """单层交叉网络"""
    def __init__(self, input_dim ,rank = 10):
        super().__init__()
        #self.weight = nn.Parameter(torch.randn(input_dim, 1))  # W: (d, 1)
        self.U = nn.Parameter(torch.randn(input_dim, rank) * 0.01)  # U: (d, r)
        self.V = nn.Parameter(torch.randn(input_dim, rank) * 0.01)  # V: (d, r)
        self.bias = nn.Parameter(torch.zeros(input_dim))        # b: (d,)

    def forward(self, x0, xi):
        # x0: 初始输入 (batch_size, input_dim)
        # xi: 当前层输入 (batch_size, input_dim)
        #linear_out = torch.matmul(xi, self.weight).squeeze() + self.bias  # (batch_size, input_dim)

        xv = torch.matmul(xi, self.V)  # (batch_size, rank)
        uv = torch.matmul(xv, self.U.t()) + self.bias # (batch_size, input_dim)
        return x0 * uv + xi  # Hadamard积 + 残差连接

class CrossNetwork(nn.Module):
    """多层交叉网络（堆叠多个CrossLayer）"""
    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossLayer(input_dim) for _ in range(num_layers)
        ])
    def forward(self, x0):
        xi = x0  # 初始输入
        for layer in self.layers:
            xi = layer(x0, xi)  # 每层计算: x_{i+1} = x0 ⊙ (W_i × x_i + b_i) + x_i
        return xi  # 最终输出

class DeepNetwork(nn.Module):
    """
    DNN部分，用于提取高阶特征
    """
    def __init__(self, input_dim, hidden_dims, dropout=0.1):
        """
        :param input_dim: 输入特征维度
        :param hidden_dims: 隐藏层维度列表，例如[256, 128, 64]
        :param dropout: dropout比率
        """
        super(DeepNetwork, self).__init__()
        layers = []

        # 构建多层感知机
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            input_dim = hidden_dim

        self.dnn = nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: 输入特征, shape为(batch_size, input_dim)
        :return: DNN输出, shape为(batch_size, hidden_dims[-1])
        """
        return self.dnn(x)


class DCN(nn.Module):
    """
    Deep & Cross Network V2模型
    """
    def __init__(self, input_dim, cross_layers, deep_layers, output_dim=1):
        """
        :param input_dim: 输入特征维度
        :param cross_layers: cross network的层数
        :param deep_layers: deep network的隐藏层维度，例如[256, 128]
        :param output_dim: 输出维度，二分类为1
        """
        super(DCN, self).__init__()

        self.cross_network = CrossNetwork(input_dim, cross_layers)
        self.deep_network = DeepNetwork(input_dim, deep_layers)

        # 组合层，连接cross和deep的输出
        concat_dim = input_dim + deep_layers[-1]  # Cross网络输出维度 + Deep网络输出维度
        self.final_layer = nn.Linear(concat_dim, output_dim)

    def forward(self, x):
        """
        :param x: 输入特征，shape为(batch_size, input_dim)
        :return: 模型输出，对于二分类问题，shape为(batch_size, 1)
        """
        # 获取Cross和Deep的输出
        cross_output = self.cross_network(x)  # (batch_size, input_dim)
        deep_output = self.deep_network(x)  # (batch_size, deep_layers[-1])

        # 连接两个输出
        concat_output = torch.cat([cross_output, deep_output], dim=1)  # (batch_size, input_dim + deep_layers[-1])

        # 最终输出层
        output = self.final_layer(concat_output)

        return output


class DCNExpert(nn.Module):
    """作为专家网络的DCN模型"""
    def __init__(self, input_dim,output_dim, cross_layers=2, deep_layers=[128, 64]):
        super(DCNExpert, self).__init__()

        self.cross_network = CrossNetwork(input_dim, cross_layers)
        self.deep_network = DeepNetwork(input_dim, deep_layers)

        # 组合层，连接cross和deep的输出
        self.concat_dim = input_dim + deep_layers[-1]
        self.final_layer = nn.Linear(self.concat_dim, output_dim)  # 输出维度与输入相同，以便在MMoE中组合

    def forward(self, x):
        # 获取Cross和Deep的输出
        cross_output = self.cross_network(x)  # (batch_size, input_dim)
        deep_output = self.deep_network(x)  # (batch_size, deep_layers[-1])

        # 连接两个输出
        concat_output = torch.cat([cross_output, deep_output], dim=1)

        # 最终输出层
        return self.final_layer(concat_output)


# 使用示例
if __name__ == "__main__":
    # 参数设置
    batch_size = 64
    input_dim = 40  # 输入特征维度
    cross_layers = 3  # Cross网络层数
    deep_layers = [128, 64]  # Deep网络隐藏层

    # 创建随机输入
    x = torch.randn(batch_size, input_dim)

    # 初始化模型
    model = DCN(input_dim, cross_layers, deep_layers)

    # 前向传播
    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")

    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params}")