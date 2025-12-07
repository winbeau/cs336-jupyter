```python
import torch
import import_ipynb
from TransformerLM import TransformerLM
from BPE_Tokenizer import Tokenizer
from TrainFunctions import load_checkpoint
```

    ./TrainLoopFiles/checkpoints



```python
device = "cuda" if torch.cuda.is_available() else "cpu"

tok = Tokenizer.from_files( # 加载 tokenizer
    "./bpeModel/vocab.json",
    "./bpeModel/merges.txt",
    special_tokens=["<|endoftext|>"]
)
ckpt_path = "./TrainLoopFiles/checkpoints_localhost/checkpoint_final.pt"

# 模型超参
vocab_size=10000 
num_layers=6
d_model=512
num_heads=8
max_seq_len=512

# 初始化模型并加载权重
model = TransformerLM(vocab_size, d_model, num_heads, num_layers, max_seq_len)

load_checkpoint(model, optimizer=None, src_path=ckpt_path)
model.to(device)
```




    TransformerLM(
      (token_emb): Embedding()
      (blocks): ModuleList(
        (0-5): 6 x TransformerBlock(
          (norm1): RMSNorm()
          (norm2): RMSNorm()
          (attn): MultiHeadSelfAttention(
            (W_Q): Linear()
            (W_K): Linear()
            (W_V): Linear()
            (W_O): Linear()
            (rope): RoPE()
          )
          (ffn): SwiGLU(
            (W1): Linear()
            (W2): Linear()
            (W3): Linear()
          )
        )
      )
      (norm): RMSNorm()
      (out): Linear()
    )




```python
@torch.no_grad()
def generate_text(
    model, 
    tokenizer, 
    prompt, 
    max_new_tokens=128, 
    temperature=0.8, 
    top_p=0.9
):
    model.eval()
    device = next(model.parameters()).device # 自动模型所在检测设备
    ids = tokenizer.encode(prompt)
    if len(ids) == 0:
        ids = [0]
    tokens = torch.tensor([ids], device=device)  # shape [1, T]

    for _ in range(max_new_tokens):
        logits = model(tokens)[:, -1, :] # [b, t, v] 对 t 只取最后一个位置的预测
        """数值清理"""
        logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
        logits = torch.clamp(logits, -20, 20) # 限制极值
        probs = torch.softmax(logits / temperature, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0)

        """top-p 采样"""
        sorted_p, sorted_i = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_p, dim=-1) # 累计概率超过 top_p 时截断
        mask = cum > top_p
        sorted_p[mask] = 0

        if sorted_p.sum() <= 0 or not torch.isfinite(sorted_p).all():
            next_id = torch.argmax(probs, dim=-1) # 和 0 或 NaN 则贪心选择
        else:
            sorted_p /= sorted_p.sum() # 按概率分布随机采样
            sampled = torch.multinomial(sorted_p, 1)
            next_id = sorted_i.gather(-1, sampled).squeeze(1)

        """拼接修正"""
        tokens = torch.cat([tokens, next_id.unsqueeze(1)], dim=1)

        if next_id.item() == tokenizer.token_to_id("<|endoftext|>"):
            break

    return tokenizer.decode(tokens[0].tolist())
```


```python
text = generate_text(model, tok, "Once upon a time", max_new_tokens=256)
print("=== Generated Text ===")
print(text)
```

    === Generated Text ===
    Once upon a time, there was a little girl named Sam. They loved to clean in the park. They looked at the door and helped it. She started to play with her ball.  They were the fire toy different and saw a big home. They had fun on the floor, and saw the wanted to do the ra



```python

```
