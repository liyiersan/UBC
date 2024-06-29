import sys
sys.path.append('./')
import torch
import torch.nn as nn

from models.base_model import BaseModel
from models.common import LinearEmbedding, PositionalEncoding


class Transformer(BaseModel):
    """
    用于时序数据的Transformer模型

    Parameters:
        in_ch (int): 输入数据的通道数
        num_layers (int): Transformer中的encoder层数
        d_model (int): Transformer中的embedding维度
        nhead (int): 多头注意力机制的头数
        dim_ffn (int): FeedForward层的维度
        dropout (float): Dropout概率
        feat_dim (int): 提取的特征维度
        num_classes (int): 分类的类别数
        norm_linear (bool): 是否使用归一化的线性层作为分类器
    """

    def __init__(self, in_ch = 13, num_layers= 3,  d_model = 128, nhead = 8, dim_ffn = 512, dropout = 0.1, feat_dim = 128,  num_classes=6, norm_linear=False):
        super(Transformer, self).__init__(feat_dim, num_classes, norm_linear)
        # Embedding layer that converts input tokens to vectors of a specified size
        self.embedding = LinearEmbedding(in_ch, d_model)

        # Positional encoding layer for the transformer
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Transformer Encoder consisting of several layers of TransformerEncoderLayer
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_ffn, dropout, activation='relu', batch_first=True),
            num_layers
        )

    def forward(self, x):
        x = self.embedding(x) 
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = torch.mean(x, dim=1) # [B, N, C] -> [B, C]
       
        return self.linear(x), x 


if __name__ == '__main__':
    model = Transformer()
    print(model)
    input = torch.randn(100, 64, 13)
    output = model(input)
    print(output[0].shape)