# 来自《Attention is All You Need》的Scaled Dot-Product Attention
# 教程：https://zhuanlan.zhihu.com/p/665148654
# 序列编码的基本思路有多种：
# 第一种是RNN，递归式进行，结构比较简单但是无法并行计算，速度较慢
# 第二种是CNN，使用窗口进行遍历，虽然很方便并行但难以捕获长距离依赖，且并行化程度不够，此外CNN对位置信息不敏感，因为卷及操作具有平移不变性，添加位置信息不如Transformer方便
# 第三种就是Transformer，以往要获得全局信息要么用双向的RNN，要么需要堆叠CNN，但是Transformer可以直接获得全局信息，且可以并行计算，速度快，效果好
# Transformer的缺点是需要大量的计算资源，因为要计算所有的位置对之间的attention，这个计算量是平方级别的，但是可以通过一些方法进行优化，比如对一部分位置添加mask
# Transformer主要包含三部分：用于编码的Encoder，用于解码的Decoder，以及输出层generator

import torch
import torch.nn as nn

# 先设置一些模型的参数
d_model = 512  # 词向量长度
d_ffn = 2048  # 隐藏层维度
d_k = d_v = 64  # qkv的词向量长度，d_q=d_k
n_layers = 6  # encoder layers, decoder layers 的个数
n_heads = 8  # 注意力的头数


# 正弦位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model=512, dropout=0.1, max_length=5000):
        """
        :param d_model
        :param dropout
        :param max_length: default=5000,假设一个句子最多包含5000个token
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # 接下来进行位置编码，首先生成一个 max_len * d_model 大小的矩阵，能覆盖整个输入的词和对应嵌入向量
        PE = torch.zeros(max_length, d_model)  # PE.shape = [5000, 512]
        pos = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)  # pos.shape = [5000,1]
        # 正弦位置编码公式：偶数项：sin(k/(10000^(2i/d))),奇数项换cos,其中：k是位置,i是分量标号,d是位置向量维度（分量总数）
        # 先求出括号内项 temp 广播机制：[5000,1]/[256] = [5000,256]/[5000,256] = [5000,256]
        temp = pos / pow(10000, torch.arange(0, d_model, 2).float() / d_model)
        PE[:, 0::2] = torch.sin(temp)  # 偶数项取sin
        PE[:, 1::2] = torch.cos(temp)  # 奇数项取cos
        # 每个句子都要做一次PE，所以在第一个维度加一个batch，维度变成[1,5000,512]
        PE = PE.unsqueeze(0)
        # 将PE放入缓冲区，将不会在训练中更新
        self.register_buffer('PE', PE)

    def forward(self, x):
        """输入x，已经预先准备了最大长度的PE，按照需要取出并和输入相加即可
        Args:
            x:[batch_size, seq_len, d_model]
        Return:
            dropout(x):[batch_size, seq_len, d_model]
        """
        x = x + self.PE[:, :x.size(1), :]  # 此处可以使用self.PE是因为register_buffer()可以将储存的变量变成元素
        return self.dropout(x)


# 补齐部分掩码 Padding Mask
# 因为输入的序列长短不一，所以需要用占位符<PAD>都补足成为最长句子的长度，而这些补齐位置不需要参加注意力的计算，所以添加掩码
# 当mask=True时遮掩，在计算出注意力分数矩阵后对该矩阵作用，注意力分数矩阵的(i,j)表示第i个词的q与第j个词的k的注意力分数，而占位符不应该被查询，此时j列应该被mask
# 为什么只遮掩k不遮掩q？
def get_attn_pad_mask(seq_q: torch.Tensor, seq_k: torch.Tensor) -> torch.Tensor:
    """
    为encoder_input和decoder_input做一个mask，把占位符mask掉
    Args:
        seq_q:
        seq_k:

    Returns:
        bool:[batch_size, len_q, len_k]
    """
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    attn_pad_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k] 1用于广播
    return attn_pad_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]





