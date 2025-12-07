```python
import torch
from dataclasses import dataclass, field
from typing import Dict, Tuple, Iterable
import math, os
import numpy as np
```

## 1-交叉熵 | 数值稳定
- 交叉熵(Cross Entropy Loss) 在所有 Transformer Block 训练之后，对 其 `[batch_size, seq_len, vocab_size]` 的未归一化分数进行计算损失

- 模型是为了预测下一个 token 故输出张量最后一维 softmax 后才是真正的模型结果

- 函数语义: 求 真值位置的 soft 得分 在 所有位置 soft 得分之和的比例，再取对数、负数

- 真值得分越高，占比越大，取对数越大，取负数后越小 -> 预测错误损失会越大

> 对于每个 batch 样本，模型输出形状是`logits: (seq_len, vocab_size)`
> 对序列中每个位置，取该位置对应的 `logits[i]`(一个长度为 vocab_size 的向量)，
> 经过 softmax 得到一个分布即序列中 下一个token 的概率分布

### 如何做到数值稳定?
设最后一维每 vocab 每个 token 得分向量为 sc，真值 y，则对应每一个 seq_len 的 i 位置: 

$$
loss = -log\frac{e^{sc_y}}{\sum_j^Ve^{sc_j}} = log\sum_j^Ve^{sc_j} - sc_y
$$

数值稳定问题? 分母求和可能会爆掉，尝试对所有元素减去最大值，softmax单调性不变
$$
log\sum_j^Ve^{sc_j} = log\sum_j^Ve^{sc_j - sc_m} + sc_m
$$


```python
def cross_entropy_loss(logits, targets): # logits[b, l, v] 推理结果 targets[b, l] 目标结果
    logits_f32 = logits.to(torch.float32)
    
    max_logits, _ = torch.max(logits_f32, dim=-1, keepdim=True) # [b, l, v] -> [n, l, 1]
    shifted = logits_f32 - max_logits # 广播机制 [b, l, v]
    # 支持任意批维度
    sum_exp = torch.sum(torch.exp(shifted), dim=-1, keepdim=False) # [b, l] 作为 seq_len 的每一token预测的soft总分
    log_sum_exp = torch.log(sum_exp) + max_logits.squeeze(-1) # max_logits: [b, l, 1] -> [b, l]

    # targets.unsqueeze(-1) [b, l] -> [b, l, 1]  logits: [b, l, v] take_along_dim ->[b, l, 1]    
    true_logits = torch.take_along_dim(logits_f32, targets.unsqueeze(-1), dim=-1).squeeze(-1) # [b, l]

    tmp = log_sum_exp - true_logits
    return tmp.mean() # 对所有元素求平均
```

## 2-AdamW | Adam + 权重衰退

随机梯度下降 (SGD) 存在如下问题:
- 若梯度波动较大(噪声), 梯度下降的过程可能会在一个 "山谷" 来回波动
- 梯度较大时，参数更新的步长很大，容易忽略掉很多信息；梯度较小时，参数更新的步长很小，效率变慢/时间被拖长
  不同的参数的梯度不同，参数更新不同步
- 缺少正则化操作容易过拟合

### Adam 如何做?

- 动量法: 记录之前的梯度值，用超参数 $\beta$ 调整当前梯度的权重，为梯度下降增加 "惯性" 约束
- 学习率自适应: 通过计算平方梯度的平均值(动量法下的梯度均方)来对学习率进行一定的缩放:
  梯度较大，梯度均方较大，学习率缩小；梯度较小，梯度均方较大，学习率增大

### 权重衰退如何做?

### AdamW 具体实现中的语义

1. 动量法:
    - $m_t = \beta_1 m_{t-1} + (1 - \beta_1)g_t ~~~ | ~~~ \text{Expoential Moving Average}$ -> `exp_avg`
      控制梯度下降方向
    - $v_t = \beta_2v_{t-1} + (1 - \beta_2)g_t^2 ~~~ | ~~~ \text{Exponential Average of Squared Gradients}$ -> `exp_avg_sq`
      控制梯度下降幅度，选平方平均是为了去线性
    - $\text{time step}$ -> `step`

> ps:
> 前几步进行滑动平均计算动量时 会出现 bias bias 问题(初始为0，前几步平均值偏小)
>
> Adam 做修正(思想: 初始时扩大，次数累计后可忽略，并且是一个平滑的过渡):
> 
> $\hat{m_t} = \frac{m_t}{1 - \beta_1^t}$， $\hat{v_t} = \frac{v_t}{1 - \beta_2^t}$
> 
2. 学习率自适应 $\alpha_t \frac{1}{\sqrt{\hat{v_t}} + \epsilon}$
3. **计算过程优化**:

> 一阶动量(`exp_avg`)控制方向 ✨ 平滑修正 $\frac{1}{1 - \beta_1^t}$、$\frac{1}{1 - \beta_2^t}$ ✨ 二阶动量(`exp_avg_sq`适应学习率)
> $$
\theta_t = \theta_{t - 1} - \alpha \cdot \frac{\frac{m_t}{1 - \beta_1^t}}{\sqrt{\frac{v_t}{1 - \beta_2^t}} + \epsilon}
$$
> 整理得：
> $$
\frac{\frac{m_t}{1 - \beta_1^t}}{\sqrt{\frac{v_t}{1 - \beta_2^t}} + \epsilon}
= \frac{m_t}{\sqrt{v_t} + \epsilon \sqrt{1 - \beta_2^t}} \cdot 
\frac{\sqrt{1 - \beta_2^t}}{1 - \beta_1^t}
$$
> 得到偏执修正系数 $\frac{\sqrt{1 - \beta_2^t}}{1 - \beta_1^t}$


```python
class AdamW(torch.optim.Optimizer):  
    def __init__(
        self, 
        params, 
        lr: float = 1e-3, 
        betas: Tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8, 
        weight_decay=0.0
    ): # betas 动量超参
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0:
            raise ValueError(f"Invalid eps: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay) # 默认参数
        super().__init__(params, defaults) # torch.optim.Optimizer 会自创建 self.param_groups

    @torch.no_grad() # 禁止梯度追踪
    def step(self, closure=None):
        loss = None
        if closure is not None: # closure 函数重新计算 loss + backward
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups: # 遍历所有组 每一组是一层的超参数
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]: # group["params"] : 需要被优化的参数(权重)
                if p.grad is None: 
                    continue 
                grad = p.grad.data
                state = self.state[p] # 提取当前操作参数(权重)进行计算
                if len(state) == 0: # 初始化动量参数
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data) 
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                state["step"] += 1
                t = state["step"]

                exp_avg.mul_(beta1).add_(grad, alpha=(1 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=(1 - beta2))

                bias_cor1 = 1 - beta1 ** t
                bias_cor2 = 1 - beta2 ** t 

                step_size = lr * math.sqrt(bias_cor2) / bias_cor1 # 实际步长 | 简化除法计算

                denom = exp_avg_sq.sqrt().add_(eps) # 归一化计算滑动平均 v_t | denominator 分母

                p.data.addcdiv_(exp_avg, denom, value=-step_size) 

                if wd != 0: 
                    p.data.add_(p.data, alpha=-lr * wd) # 超参数 wd: 让所有参数每步都向 0 缩一点
        return loss
```

## 3-余弦退火
```python
import matplotlib.pyplot as plt
import math

lrs = [cosine_lr_schedule(t, 1e-3, 1e-5, 500, 5000) for t in range(6000)]
plt.plot(lrs)
plt.title("Cosine Annealing LR Schedule with Warmup")
plt.xlabel("Iteration")
plt.ylabel("Learning Rate")
plt.show()
```


```python
def cosine_lr_schedule(
    t: int,
    lr_max: float, 
    lr_min: float, 
    warmup_iters: int, 
    cosine_iters: int,
) -> float: 
    if t < warmup_iters: # 线性预热
        return lr_max * (t / max(1, warmup_iters))
    if t <= cosine_iters: # 余弦退火
        progress = (t - warmup_iters) / max(1, (cosine_iters - warmup_iters))
        cosine_term = 0.5 * (1.0 + math.cos(math.pi * progress)) # 上平移 1.0 乘 0.5 归一化
        return lr_min + (lr_max - lr_min) * cosine_term
    return lr_min # 退火结束 保持最小学习率   

# import matplotlib.pyplot as plt
# from matplotlib_inline.backend_inline import set_matplotlib_formats

# set_matplotlib_formats('svg')  # svg 绘制

# lrs = [cosine_lr_schedule(t, 1e-3, 1e-5, 500, 5000) for t in range(6000)]
# plt.plot(lrs)
# plt.title("Cosine Annealing LR Schedule with Warmup")
# plt.xlabel("Iteration")
# plt.ylabel("Learning Rate")
# plt.show()
```

## 4-梯度裁剪
通过计算全局L2范数 若它大于 max_norm 则按比例缩放，使新范数不超过 max_norm

```python
loss = cross_entropy_loss(...)
loss.backward()           # 1️-生成梯度
clip_gradients(params)    # 2-梯度裁剪（修正梯度）
optimizer.step()          # 3️-使用被裁剪的梯度更新参数
```


```python
@torch.no_grad()
def clip_gradients(
    params: Iterable[torch.nn.Parameter], 
    max_norm: float, 
    eps: float = 1e-6
): 
    total_norm_sq = 0.0 
    for p in params: # 计算 L2 范数
        if p.grad is not None: 
            total_norm_sq += float(torch.sum(p.grad.data.to(torch.float32) ** 2))
    total_norm = math.sqrt(total_norm_sq)

    if total_norm > max_norm: 
        scale = max_norm / (total_norm + eps)
        for p in params: 
            if p.grad is not None: 
                p.grad.data.mul_(scale)

    return total_norm
```

## 5-随机采样

2.2 M 个120-150词的小故事，随机截取连续 token 片段会把一个完整得小故事切断，但不会对模型训练造成问题

`get_batch` 随机截取时:
- 片段可能刚好落在故事中
- 可能会跨过  `<|endoftext|>`
- 模型会学习到——`<|endoftext|>` 是故事结尾
### 借助 NumPy 库切片(左闭右开)
x\[i : i + m\] -> input <br>
x\[i + 1 : i + 1 + m\] -> output

### 采用随机采样 而不是 epoch 式遍历
- 不需要记录位置或状态；
- 每步梯度更新都独立抽样；
- 与随机梯度下降（SGD / AdamW）的假设完全一致。


```python
def get_batch(data, batch_size, context_len, device="cuda"): 
    n = len(data) 
    starts = np.random.randint(0, n - context_len - 1, size=(batch_size,))
    # 由于预测下一个 token  取 n - context_len - 1 位置作为样本起点

    x_batch = np.stack([data[i : i + context_len] for i in starts])
    y_batch = np.stack([data[i + 1 : i + 1 + context_len] for i in starts])

    input_batch = torch.tensor(x_batch, dtype=torch.long, device=device)
    output_batch = torch.tensor(y_batch, dtype=torch.long, device=device)
    return input_batch, output_batch
```

## 6-checkpoint 保存/读取
- 保存模型权重、优化器状态、当前 step
- 可以从 checkpoint 恢复训练


```python
def save_checkpoint(model, optimizer, iteration: int, out_path: str):
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(payload, out_path)

def load_checkpoint(model, optimizer, src_path: str) -> int:
    payload = torch.load(src_path, map_location="cpu", weights_only=True) # 确保即使保存用 GPU，CPU 环境依然可加载
    model.load_state_dict(payload["model_state"])
    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    iteration = payload.get("iteration", 0) # 默认 0 
    return iteration
```

## 7-加载数据集


```python
def load_token_dataset(path: str, mmap: bool = True):
    if mmap: # 使用内存映射模式，只在访问时加载数据
        arr = np.load(path, mmap_mode="r")
    else:
        arr = np.load(path)
    print(f"[load_token_dataset] Loaded {path}, shape={arr.shape}, dtype={arr.dtype}")
    return arr
```

## 8-TrainingConfig 集中管理超参


```python
@dataclass
class TrainingConfig:
    # === Data / batching ===
    batch_size: int = 32
    context_len: int = 256
    vocab_size: int = 10000  # tokenizer vocab size

    # === Model architecture ===
    num_layers: int = 12
    num_heads: int = 16
    d_model: int = 1024
    d_ff: int = field(default=None)  # 默认按 8/3*d_model 计算
    rope_theta: float = 10000.0

    # === Training steps ===
    total_steps: int = 10_000
    log_every: int = 50
    eval_every: int = 500
    ckpt_every: int = 1000

    # === Optimizer (AdamW) ===
    lr_max: float = 3e-4
    lr_min: float = 3e-5
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.1
    grad_clip_norm: float = 1.0

    # === Scheduler ===
    warmup_iters: int = 200
    cosine_iters: int = 10000  # after this, lr = lr_min

    # === Misc ===
    device: str = "cuda"
    target_dir = "./TrainLoopFiles"
    ckpt_dir: str = None
    ckpt_save_prefix: str = "checkpoint"
    log_dir: str = None
    log_save_prefix: str = "train"
    ckpt_path_to_resume: str = None

    # === Derived attributes ===
    def __post_init__(self): # 自动计算 d_ff, (8/3 × d_model, 且为64的倍数)
        if self.d_ff is None:
            multiple = 64
            raw_dff = int((8 / 3) * self.d_model)
            self.d_ff = multiple * ((raw_dff + multiple - 1) // multiple)
        if self.target_dir is None:
            self.target_dir = "."
        if self.ckpt_dir is None: 
            self.ckpt_dir = os.path.join(self.target_dir, "checkpoints")
        if self.log_dir is None: 
            self.log_dir = os.path.join(self.target_dir, "logs")
    
    @staticmethod # 静态成员函数
    def cal_d_ff(d_m) -> int: # 计算 d_ff
        multiple = 64
        raw_dff = int((8 / 3) * d_m)
        return multiple * ((raw_dff + multiple - 1) // multiple)
```


```python
cfg = TrainingConfig(ckpt_dir="./TrainLoopFiles/test")
```


```python
cfg.ckpt_dir
```




    './TrainLoopFiles/test'




```python
cfg1 = TrainingConfig()
```


```python
print(cfg1.ckpt_dir)
```

    ./TrainLoopFiles/checkpoints



```python

```
