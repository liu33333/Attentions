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
import numpy as np
from torch.nn import functional as F

# 先设置一些模型的参数
d_model = 512  # 词向量长度
d_ffn = 2048  # 隐藏层维度
d_k = d_v = 64  # qkv的词向量长度，d_q=d_k
n_layers = 6  # encoder layers, decoder layers 的个数
n_heads = 8  # 注意力的头数


# 数据预处理
sentence = [
    # enc_input   dec_input    dec_output
    ['ich mochte ein bier P', 'S i want a beer .', 'i want a beer . E'],
    ['ich mochte ein cola P', 'S i want a coke .', 'i want a coke . E'],
]

# 词典，padding用0来表示
# 源词典
src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4, 'cola': 5}
src_vocab_size = len(src_vocab)  # 6
# 目标词典（包含特殊符）
tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'coke': 5, 'S': 6, 'E': 7, '.': 8}
# 反向映射词典，idx ——> word
idx2word = {v: k for k, v in tgt_vocab.items()}
tgt_vocab_size = len(tgt_vocab)  # 9

src_len = 5  # 输入序列enc_input的最长序列长度，其实就是最长的那句话的token数
tgt_len = 6  # 输出序列dec_input/dec_output的最长序列长度


# 构建模型输入的Tensor
def make_data(sentence):
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    for i in range(len(sentence)):
        enc_input = [src_vocab[word] for word in sentence[i][0].split()]
        dec_input = [tgt_vocab[word] for word in sentence[i][1].split()]
        dec_output = [tgt_vocab[word] for word in sentence[i][2].split()]

        enc_inputs.append(enc_input)
        dec_inputs.append(dec_input)
        dec_outputs.append(dec_output)

    # LongTensor是专用于存储整型的，Tensor则可以存浮点、整数、bool等多种类型
    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)


enc_inputs, dec_inputs, dec_outputs = make_data(sentence)

print(' enc_inputs: \n', enc_inputs)  # enc_inputs: [2,5]
print(' dec_inputs: \n', dec_inputs)  # dec_inputs: [2,6]
print(' dec_outputs: \n', dec_outputs)  # dec_outputs: [2,6]





# 1. 正弦位置编码
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

# 2. 补齐部分掩码 Padding Mask
# 因为输入的序列长短不一，所以需要用占位符<PAD>都补足成为最长句子的长度，而这些补齐位置不需要参加注意力的计算，所以添加掩码
# 当mask=True时遮掩，在计算出注意力分数矩阵后对该矩阵作用，注意力分数矩阵的(i,j)表示第i个词的q与第j个词的k的注意力分数，而占位符不应该被查询，此时j列应该被mask
# 为什么只遮掩k不遮掩q？ 1.防止占位符干扰其他注意力计算，所以必须让占位符无法被查询，k必须被遮掩 2.为了让模型自己学到处理占位符的方法，所以没必要遮掩q
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


# 3. 后续序列掩码 Subsequence Mask
# 此模块对应的是Decoder中的mask，用于遮掩后续序列防止模型看到未来时刻的输入，当前被预测的token之后的都需要mask掉
def get_attn_subsequence_mask(seq:torch.Tensor):
    """ 得到一个对后续位置的掩码，是batch个tgt_len*tgt_len的矩阵，右上三角矩阵，值为1的位置需要mask掉，如token1行的token2、3、4...列需要mask
    Args:
        seq: [batch_size, tgt_len]

    Returns:
        subsequence_mask: [batch_size, tgt_len, tgt_len]
    """
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape),k=1) # 得到了batch_size个形状为t_l*t_l的上三角矩阵
    subsequence_mask = torch.from_numpy(subsequence_mask).byte() # 因为只有0和1，所以转换为byte（8位无符号整数，0~255）节约内存
    return subsequence_mask


# 4. 缩放点积注意力 ScaledDotProduct Attention
# 此模块用于计算缩放点积注意力，在多头注意力中被调用
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, attn_mask:torch.Tensor):
        """
        Args:
            Q: [batch_size, n_heads, len_q, d_k]
            K: [batch_size, n_heads, len_k, d_k]
            V: [batch_size, n_heads, len_v, d_v]
            attn_mask: [batch_size, n_heads, seq_len, seq_len]

        Returns:
            context: [batch_size, n_heads, len_q, d_v]
        """
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-1. -2)) / np.sqrt(d_k) # scores: [batch_size, n_heads, len_q, len_k]

        # 计算mask和Softmax
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores) # attn: [batch_size, n_heads, len_q, len_k]

        # 乘v得到最终的加权和
        context = torch.matmul(attn, V) # context: [batch_size, n_heads, len_q, d_v]

        return  context

# 5. 多头注意力 MultiHead Attention
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.concat = nn.Linear(d_model, d_model)

    def forward(self, Q:torch.Tensor, K:torch.Tensor, V:torch.Tensor, attn_mask:torch.Tensor):
        """

        Args:
            Q: [batch_size, seq_len, d_model]
            K: [batch_size, seq_len, d_model]
            V: [batch_size, seq_len, d_model]
            attn_mask: [batch_size, seq_len, seq_len]

        Returns:

        """
        batch_size = Q.size(0)
        Q = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        K = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        V = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)

        # 计算注意力
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        context = ScaledDotProductAttention()(Q, K, V, attn_mask) # context: [batch_size, n_heads, len_q, d_v]

        # concat
        context = context.transpose(1, 2).reshape(batch_size, -1, d_model) # context: [batch_size, seq_len, d_model]
        output = self.concat(context)

        return output

# 6. Feed Forward / Add & Norm
class PositionwiseFeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.FC = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.ReLU(),
            nn.Linear(d_ffn, d_model)
        )

    def forward(self, inputs):
        """
        Args:
            inputs: [batch_size, seq_len, d_model]
        Returns:

        """
        return F.layer_norm(self.FC(inputs)+inputs, normalized_shape=[d_model])

# 7. Encoder Layer
class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_MHA = MultiHeadAttention()
        self.pos_ffn = PositionwiseFeedForward()

    def forward(self, enc_input, enc_self_attn_mask):
        """
        Args:
            enc_input: [batch_size, src_len, d_model]
            enc_self_attn_mask: [batch_size, src_len, src_len]

        Returns:

        """
        enc_outputs = self.enc_MHA(enc_input, enc_input, enc_input, enc_self_attn_mask) # enc_outputs: [batch_size, seq_len, d_model]
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs # enc_outputs: [batch_size, src_len, d_model]

# 8. Encoder
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        """
        Args:
            enc_inputs: [batch_size, src_len]

        Returns:

        """
        enc_outputs = self.src_emb(enc_inputs) # [batch_size, src_len] -> [batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_outputs)
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs) # enc_self_attn_mask: [batch_size, src_len(len_q), src_len(len_k)]
        for layer in self.layers:
            enc_outputs =layer(enc_inputs, enc_self_attn_mask)
        return enc_outputs
