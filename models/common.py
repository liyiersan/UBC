import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class NormedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out

class ResidualBlock(nn.Module):
    """
    ResNet的基本构建块, 包含两个卷积层和一个残差连接. 

    Parameters:
    in_ch (int): 输入通道数
    out_ch (int): 输出通道数
    stride (int, optional): 卷积层的步长, 默认为1.
    shortcut (nn.Module, optional): 残差连接的分支操作, 如果为空, 则采用恒等映射.
    """
    def __init__(self, in_ch, out_ch, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, stride, 1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm1d(out_ch)
        )
        self.right = nn.Identity() if shortcut is None else shortcut

    def forward(self, x):
        out = self.left(x)
        residual = self.right(x)
        out += residual
        return F.relu(out)

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len) -> None:
        super(LearnablePositionalEncoding, self).__init__()
        self.pe = nn.Parameter(torch.randn(1, seq_len, d_model))
    
    def forward(self, x):
        x_len = x.size(1)
        return x + self.pe[:, :x_len, :]

class PositionalEncoding(nn.Module):
    """
    Transformer的位置编码模块.

    Parameters:
    d_model (int): 位置编码的特征维度.
    dropout (float): Dropout比例.
    max_len (int): 所编码的序列最大长度.
    """

    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        # 计算位置编码
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # dim 2i
        pe[:, 1::2] = torch.cos(position * div_term)  # dim 2i+1
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        # register_buffer()用于在模型中注册一些常量值. 它将成为模型的state_dict()的一部分, 但不会被训练
        self.register_buffer('pe', pe) 

    def forward(self, x):
        """
        Parameters:
            x (torch.Tensor): 输入时序数据,  (B, N, C).

        Returns:
            torch.Tensor: 带有位置编码的序列输入, (B, N, C).
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TokenEmbedding(nn.Module):
    """
        Transformer的token embedding模块, 它将输入数据转换成维度为d_model的embedding向量

    """
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        """
        Parameters:
            x (torch.Tensor): 数据时序数据, (B, N, C).
        return (torch.Tensor): embedding, (B, N, d_model).
        """
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x
    
class LinearEmbedding(nn.Module):
    def __init__(self, c_in, d_model, activation=None):
        super(LinearEmbedding, self).__init__()
        self.linear = nn.Linear(c_in, d_model)
        self.activation = activation

    def forward(self, x):
        x = self.linear(x)
        if self.activation:
            x = self.activation(x)
        return x
    
class ConvEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(ConvEmbedding, self).__init__()
        self.conv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=1, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')
        
    def forward(self, x):
        return self.conv(x.permute(0, 2, 1)).transpose(1, 2)
    
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class SelfAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class PositionalEncoding(nn.Module):
    """
    Transformer的位置编码模块.

    Parameters:
    d_model (int): 位置编码的特征维度.
    dropout (float): Dropout比例.
    max_len (int): 所编码的序列最大长度.
    """

    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 计算位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # dim 2i
        pe[:, 1::2] = torch.cos(position * div_term)  # dim 2i+1
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        # register_buffer()用于在模型中注册一些常量值. 它将成为模型的state_dict()的一部分, 但不会被训练
        self.register_buffer('pe', pe) 

    def forward(self, x):
        """
        Parameters:
            x (torch.Tensor): 输入序列数据,  (B, N, C).

        Returns:
            torch.Tensor: 带有位置编码的序列输入, (B, N, C).
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
    
class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(
            self,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer = None,
            bias: bool = True,
    ):
        super().__init__()

        # 卷积的步长和卷积核大小相同, 保证不重叠
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        
        x = self.proj(x) # (B, C, H, W)
        x = self.norm(x)
        # reshape to (B, embed_dim, N)
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x