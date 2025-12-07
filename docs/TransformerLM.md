# Transformer From 0 To 1
1. Embedding | 语句 -> 词向量
2. RMSNorm(Layer Normalization) | 层归一化-预归一化(pre-norm)
3. Multi-Head-Attention:
   - Linear(W_Q, W_K, W_V, W_O)
   - RoPE | 旋转-位置信息
   - softmax
4. SwiGLU(FeedForward) | 激活函数-门控模块
5. Linear | 输出层


```python
import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
```


```python
class Embedding(nn.Module): # @ Embedding weight
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None): 
        super().__init__() # 使用 empty() 初始化形状，无需填充 0 更快
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids): 
        return self.weight[token_ids] # 返回对应权重的索引行
```


```python
class RMSNorm(nn.Module): # @ RMSNorm gain
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None): 
        super().__init__() 
        self.eps = eps # 防止除 0 的小常数
        self.gain = nn.Parameter(nn.Parameter(torch.ones(d_model, device=device, dtype=dtype)))
        # gain: 缩放系数，初值为 1 不改变输入

    def forward(self, x): 
        x_dtype = x.dtype # 记录原来类型
        x = x.to(torch.float32)
        # 计算 self 在最后一维的 L2 范数
        norm = x.norm(dim=-1, keepdim=True) / (x.size(-1) ** 0.5)
        return (x / (norm + self.eps)) * self.gain.to(x_dtype)
```


```python
class Linear(nn.Module): 
    def __init__(self, d_in, d_out, device=None, dtype=None): 
        super().__init__()
        self.W = nn.Parameter(torch.empty(d_out, d_in, device=device, dtype=dtype))
        xavier_std = (2 / (d_in + d_out)) ** 0.5 # Xavier 初始化 保持in/out方差稳定性
        nn.init.trunc_normal_(self.W, mean=0.0, std=xavier_std, a=-3.0, b=3.0)

    def forward(self, x): 
        return x @ self.W.T
```

$$
\theta_{i,k} = i \cdot \frac{1}{\Theta^{2k/d_k}} ~~~~~~~
\text{其中  freq\_inv}_k = \frac{1}{\Theta^{2k/d_k}} ~~~
\Theta \, 通常取 10^4
$$

$$ 
\begin{bmatrix}
x_0' ~ x_1'
\end{bmatrix} ~ = ~ 
\begin{bmatrix}
x_0 ~~ x_1
\end{bmatrix} ~~
\begin{bmatrix}
cos \theta_k ~~ sin \theta_k \\
-sin \theta_k ~~ cos \theta_k
\end{bmatrix}
$$


```python
class RoPE(nn.Module): 
    def __init__(self, Theta, d_k, seq_len, device=None): 
        super().__init__()
        inv_freq = 1.0 / (Theta ** torch.arange(0, d_k, 2, device=device).float() / d_k) 
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.einsum("i,k->ik", t, inv_freq)
        self.register_buffer("cos", torch.cos(freqs)) # 存为 buffer 后续查表
        self.register_buffer("sin", torch.sin(freqs))

    def forward(self, x, positions): # x: [..., seq_len, d_k] position: [..., seq_len]
        cos = self.cos[positions]
        sin = self.sin[positions]
        """分隔奇偶 -> 求对应值 -> 按维度堆叠 -> 展平"""
        x0, x1 = x[..., ::2], x[..., 1::2] # 把 x 拆分为偶数序列、奇数序列
        x_rot = torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), dim = -1)
        return x_rot.flatten(-2) # 展开最后两维 
```

$$
\begin{cases}
h_1 = W_1 x \\ 
h_2 = W_3 x
\end{cases} ~~~ \Rightarrow ~~~
\tilde{h} = SiLU(h_1) \odot h_2 ~~~ \Rightarrow ~~~
y = W_2 \tilde{h}
$$


```python
class SwiGLU(nn.Module): 
    def __init__(self, d_model, mul_of=64): # multiple_of 作倍数上取整 
        super().__init__()
        d_ff = int((8/3) * d_model)
        d_ff = mul_of * ((d_ff + mul_of - 1) // mul_of)
        self.W1 = Linear(d_model, d_ff)
        self.W2 = Linear(d_ff, d_model)
        self.W3 = Linear(d_model, d_ff) 

    def forward(self, x): 
        return self.W2(F.silu(self.W1(x)) * self.W3(x))
```

### 多头自注意力 + 因果mask + RoPE
0. scaled dot-product attention 实现
1. 输入 x: (B, T, C)
2. 线性变换得到 Q_raw, K_raw, V_raw: (B, T, C)
3. reshape+transpose → (B, h, T, d_head) 得到 Q/K/V
4. 对 Q/K 用 RoPE 按位旋转，把位置信息注入
5. 构造下三角 mask (T, T)，禁止看未来
6. 对每个 head 做 scaled dot-product attention
   得到 context_per_head: (B, h, T, d_head)
7. 把多头拼回去：transpose -> view → (B, T, C)
8. 通过输出投影 W_O，仍是 (B, T, C)，交给残差/后续FFN


```python
def softmax(x, dim=-1): 
    x = x - x.max(dim=dim, keepdim=True).values
    exp_x = x.exp() 
    return exp_x / exp_x.sum(dim=dim, keepdim=True) # 广播机制

def scaled_dot_product_attention(Q, K, V, mask=None): 
    d_k = Q.size(-1) # Q: [..., seq_len, d_head]
    scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5) # K: [..., seq_len, d_head] 不能简单转置
    if mask is not None: 
        scores = scores.masked_fill(~mask, float('-inf')) # 由于要做 softmax 掩码要取反
    attn = softmax(scores, dim=-1) # 最后一个维度做 softmax
    return attn @ V    
```


```python
class MultiHeadSelfAttention(nn.Module): 
    def __init__(self, d_model, num_heads, seq_len=1024, rope_theta=10000): 
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be a multiple of num_heads"
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.W_Q = Linear(d_model, d_model)
        self.W_K = Linear(d_model, d_model)
        self.W_V = Linear(d_model, d_model)
        self.W_O = Linear(d_model, d_model)
        self.rope = RoPE(rope_theta, self.d_head, seq_len)

    def forward(self, x): 
        B, T, C = x.shape # T = seq_len C = d_model
        positions = torch.arange(T, device=x.device)
        """Q @ KT -> [..., seq_len, seq_len] -> scaled_dot -> @ V -> [..., seq_len, d_head]"""
        Q = self.W_Q(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        K = self.W_K(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        V = self.W_V(x).view(B, T, self.num_heads, self.d_head).transpose(1, 2)
        Q = self.rope(Q, positions) # RoPE: (..., seq_len, d_k) (..., seq_len)
        K = self.rope(K, positions)
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        out = scaled_dot_product_attention(Q, K, V, mask=mask) # 因果下三角
        out = out.transpose(1, 2).contiguous().view(B, T, C) # 创建连续副本 concat
        return self.W_O(out)
```


```python
class TransformerBlock(nn.Module): 
    def __init__(self, d_model, num_heads): 
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model) # 门控有参数 存两份
        self.attn = MultiHeadSelfAttention(d_model, num_heads)
        self.ffn = SwiGLU(d_model)

    def forward(self, x): # 残差连接 + 预归一化
        x = x + self.attn(self.norm1(x)) # 多头注意力
        x = x + self.ffn(self.norm2(x)) # FFN 前馈神经网络
        return x
```


```python
class TransformerLM(nn.Module): 
    def __init__(self, vocab_size, d_model, num_heads, num_layers, max_seq_len=1024): 
        super().__init__() 
        self.token_emb = Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, num_heads) for _ in range(num_layers)])
        self.norm = RMSNorm(d_model)
        self.out = Linear(d_model, vocab_size) # 线性层输出

    def forward(self, token_ids): 
        x = self.token_emb(token_ids)
        for blk in self.blocks: 
            x = blk(x)
        x = self.norm(x) # 由于 pre-norm 输出前要再进行一次 层归一化
        return self.out(x)
```


```python
# vocab_size = 1000
# model = TransformerLM(vocab_size, d_model=128, num_heads=8, num_layers=2).to(device)

# x = torch.randint(0, vocab_size, (2, 16), device=device)
# y = model(x)
# print(y.shape)  # (batch, seq_len, vocab_size)
```
