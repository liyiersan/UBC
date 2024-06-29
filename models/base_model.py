import torch.nn as nn
from models.common import NormedLinear

class BaseModel(nn.Module):
    """
        基础模型类, 所有的模型都应该继承这个类
        Parameters:
            feat_dim: 特征的维度
            num_classes: 分类的类别数
            norm_linear: 是否使用归一化的线性层
    """
    def __init__(self, feat_dim, num_classes=6, norm_linear=False):
        super(BaseModel, self).__init__()
        self.linear = NormedLinear(feat_dim, num_classes) if norm_linear else nn.Linear(feat_dim, num_classes)
        self.num_classes = num_classes
        self.feat_dim = feat_dim
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 模型的输入, shape: [batch_size, num_channels, seq_len]
        """
        return x