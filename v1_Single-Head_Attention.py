#【这份代码的前半部分就是bigram_handcode.py，但是将根据上一个token预测改成了使用单个self-attention】
# 对应视频 42:14 

import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # 一次同时处理几条字符串->next token
block_size = 8 # 最大上下文长度？（即能看见前面的几个token？）
max_iters = 5000
eval_interval = 300
learning_rate = 1e-3
eval_iters = 200
n_embd = 32
# ------------

# 此行非手搓，仅用于下载数据集
# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

#【选择GPU】
if (torch.cuda.is_available()):
    device = 'cuda'
elif (torch.backends.mps.is_available()):
    device = 'mps' # 增加对mac的支持
else:
    device = 'cpu'

#【数据处理】
# 打开文件并读取
with open("input.txt", "r", encoding="utf-8") as f: # 使用with的好处是自动关闭文件（上下文管理）
    # 从f当中读取文件内容进text
    text = f.read() 

# 提取text中出现的所有字符
chars = sorted(list(set(text))) # 先set去重，再list化，再sorted对list排序（set不可排序）

vocab_size = len(chars)

# 创造从字符串到整数索引的双向映射
# 这就是这个简单的bigram程序的tokenizer
stoi = {s:i for i,s in enumerate(chars)} # enumerate自动生成(index, value)的tuple
itos = {i:s for i,s in enumerate(chars)}

# tokenizer需要encode和decode函数（将字符串转成int list和从int list转成字符串）
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[l[i]] for i in range(len(l))]) # "".join()将list中的元素连接成一个字符串

# 将encode之后的list转换成torch当中的tensor
data = torch.tensor(encode(text), dtype=torch.long)

# 划分训练集和测试集
n = int(0.9 * len(data))
train_data = data[:n] # 这里没有将数据打乱，而是直接将前n个数据划分为训练集
validation_data = data[n:]

# LLM的精髓是根据前文预测下一个token
# 这里就是要从原始数据中，提取出这样一种用于训练的数据模式：
# 用第 [0], [0, 1], [0, 1, 2]...[0,..., block_size - 1]来预测第[1], [2], [3],..., [block_size]
def get_batch(split): # split是"train"或"validation"
    data = train_data if split == "train" else validation_data

    # 随机出batch_size个序列开始的index
    start_index = torch.randint(len(data) - block_size, (batch_size, )) # -block_size确保哪怕随机到最后一个，后面也还可以提取block_size个token
    # torch.randint(low=0, high, size,device=None) → Tensor 
    
    # 从而以这些index为起点，将预测下一个token的数据对弄出来
    # 并使用torch中的二维矩阵存储
    x = torch.stack([data[st_i:st_i + block_size] for st_i in start_index])
    y = torch.stack([data[st_i + 1:st_i + block_size + 1] for st_i in start_index])
    # torch.stack(tensors, dim=0, *, out=None) → Tensor
    # 将x和y都用stack堆叠成二维数组（应注意，计算机中的二维数组行和列的标号是和线性代数中反过来的）
    
    # 在get_batch时，一定要记得将数据搬到GPU上
    x, y = x.to(device), y.to(device)
    return x, y

#【评估loss函数】
@torch.no_grad() # 告诉torch这个函数永远不用反向传播
def estimate_loss():
    out = {}
    model.eval() # 在torch中开启评估模式，相当于一个全局开关，停止使用drop_out，BatchNorm等
    for split in ['train', 'validation']:
        losses = torch.zeros(eval_iters)
        # 在数据集中采样eval_iters次然后取平均，获得更稳定的loss估计
        for k in range(eval_iters): 
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # 评估完要调回train模式
    return out


class Head(nn.Module):
    """实现一个注意力头"""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False) # 这里key和query看起来对称，但最后一定会各司其职，想想为什么？
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        # 当将这个model搬到GPU时，register_buffer会将这个tensor也搬到GPU
        # 这里tril需要使用register_buffer，因为tril本身不更新，但需要在训练中参与计算
        # 而后续的k, q, v由原本就在GPU上的x计算出来，天然在GPU上，所以不需要register_buffer


    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        wei = q @ k.transpose(-2, -1) # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        # 高维矩阵的乘法规则是，除了最后两个维度外，前面的所有维度都看作是“批处理维度”。
        # 注意此处q一定要在k左边，不然最后q k矩阵的作用会是与变量名相反的

        # 只使用下三角矩阵，确保每个token只能跟自己和自己之前的token交流(decoder-only)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # 将上三角部分（不包括对角线）设为负无穷，这样softmax之后就会变成0； [:T, :T]切片动态适应T长度
        wei = F.softmax(wei, dim=-1) # 对每个位置i，计算它对所有位置j的注意力权重

        v = self.value(x) # (B, T, head_size)
        out = wei @ v # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out


#【模型】
# BigramModel的先验假设是：每一个token仅依赖于其相邻的上一个token
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__() # 这一行让我们能轻松地使用torch当中的各种组件（比如一键将数据搬到GPU）
        
        # 不再使用char的编号（一维）来表示一个token
        # 而是用n_embd维的向量来表示一个token
        # 这里的n_embd相当于CV中卷积的总通道数
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # n_embd是num_embedding的缩写
        
        # 由于Attention并不像卷积核一样天生带有空间信息
        # 所以我们要给每个位置用一个向量来添加位置信息
        self.position_embedding_table = nn.Embedding(block_size, n_embd) 
        # 创建block_size个不同的位置向量，每个位置一个n_embd维的向量
        # 位置0 → 向量0，位置1 → 向量1，...，位置7 → 向量7

        self.sa_head = Head(n_embd) # self-attention head
        # 创建一个线性层（作为Head），来将n_embd维的向量映射到vocab_size维的logits
        self.lm_head = nn.Linear(n_embd, vocab_size) # lm_head是language model head的缩写

    # forward函数即给定前面的字符序列，生成下一个字符的logits分布
    # 并给出loss（如果是训练状态）
    # 同时，forward函数是torch中的一个特殊函数（后续会提及）
    def forward(self, idx, targets=None): 
        B, T = idx.shape # B是Batch_size，T则借鉴了序列模型中的习惯，用Time表示block_size

        # idx是大小为(batch_size, block_size)的tensor
        # 有targets=None是为了照顾部署时，没有已知的下一个token
        
        # 先从idx中得到每个token的embedding向量（相当于是得到了每个token更丰富的信息表示）
        tok_emb = self.token_embedding_table(idx) 
        
        # 将每个位置的位置向量组合起来
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        # torch.arange(T, device=device)生成一个从0到T-1的索引列表，一次性批量提取前 T 个位置的向量。
        
        # 将token的embedding和位置的embedding相加
        x = tok_emb + pos_emb
        
        # 通过self-attention head提取特征
        x = self.sa_head(x)
        # 然后用language model Head将特征映射到vocab_size维的logits
        logits = self.lm_head(x) # logit这个词指代softmax之前的原始数值

        if targets is None:
            loss = None
        else:
            # 由于entropy的计算需要logit是2D的，
            # 即我们原本用batch和block两个维度来指代一个位置
            # 但现在我们要对它们一视同仁
            # 我们将logits展平
            batch_size, block_size, logit_num = logits.shape
            logits = logits.view(batch_size * block_size, logit_num)

            # entropy计算要求targets是一维的（即正确选项的index），所以我们也将targets展平
            targets = targets.view(batch_size * block_size)
            # Tensor.view(*shape) → Tensor 改变数据的维度
            
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    # generate函数利用forward函数，不断生成新的字符，从而输出一整段字符序列。
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # 只取最后block_size个token进行预测
            idx_cond = idx[:, -block_size:]

            logits, loss = self(idx_cond) # 这里实际上调用了forward函数，torch自动帮忙处理了很多杂活
            
            # 由于我们在BigramModel中，只需要利用到logits的最后一个token
            logits = logits[:, -1, :] # logits从(batch_size, block_size, vocab_size)变成了(batch_size, vocab_size)

            # 对logits使用softmax
            probs = F.softmax(logits, dim=-1) # dim=-1是因为最后一个维度是vocab_size（我们正是要vocab中所有token的概率分布）
            
            # 从probs分布中根据概率采样出下一个token是什么
            idx_next = torch.multinomial(probs, num_samples=1)
            # torch.multinomial(input, num_samples, ...) → LongTensor
            # 之所以叫multinomial，是因为我们在有离散个选项的概率空间中采样
            # 这就是multinomial分布（二项分布的推广）

            # 将idx_next添加到idx的末尾，供下一次生成使用
            idx = torch.cat((idx, idx_next), dim=1) # idx原形状为(batch_size, block_size), idx_next形状为(batch_size, 1), 添加了一行
        
        return idx # 现在并非流式输出

model = BigramLanguageModel()
m = model.to(device) # 创建model后也要记得将model搬到GPU上

# 不再输出未训练时的结果

# 然后我们来训练一下
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) # parameters()返回模型中所有需要优化的参数，还是torch.nn提供的便利

for step in range(max_iters):
    # 每eval_interval步输出一次当前loss
    if step % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['validation']:.4f}")

    # 获取训练数据
    xb, yb = get_batch("train")

    logits, loss = model(xb, yb) # 这一步torch会建立一个计算图，如果没有这一步backward就无法执行
    # 由于torch默认累加梯度，所以在更新梯度之前要清零上一轮的梯度
    optimizer.zero_grad(set_to_none=True) # set_to_none=True是小技巧，比设为0更省内存
    # 反向传播计算各层梯度
    loss.backward()
    # 更新各层参数
    optimizer.step()


print("训练结束后的输出：")
print(decode(model.generate(idx = torch.zeros((1, 1), dtype=torch.long, device=device), max_new_tokens=500)[0].tolist()))

# 好耶！ 
# Single-Head Attention部分到这里就算是结束了，此时loss ~ 2.4
# 我们将在另一个文件中将single-head attention 变为multi-head attention