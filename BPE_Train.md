# Byte Pair Encoding (BPE) Trainer
- 文本预分词 | special token 处理  
- 统计相邻 pair 频率 | 更新增量 
- 输出 vocab.json 与 merges.txt  

### 问题抽象
现有 $n$ 组( $n = 2 \times 10^6 $ , 每组长度约 $120$ ~ $150$ )变长字符串 $str_i$ ($len(str_i) \leq 20,~i \in [0, \, n)$)，总长度 $\sum_{i = 0}^n len(str_i) \leq 150 \times 10 \times 2 \times 10^6 = 3 \times 10^9$，

$vocab$ 为 $token ~ id$ 对 $Unicode ~ byte(s)$ 的映射集合（也就是 $token$ 集合），初始化为 $[0, 256)$ 对应其十六进制数所代表的 $Unicode ~ byte$ (可以理解为 $ASCII ~ plus$) 

**持续执行如下操作：**

对每个字符串 ***所有最长 $token$*** 进行两两结合统计频率：

<aside>

初始状态每个字符是一个 $token$

`word` → `wo` : 1  `or` : 1  `rd` : 1

> 什么是***最长 $token$*** ？如 `newest` ，`est` 已分配 $token ~ id$，我们仅对 `ne` `ew` `west` 处理
> 
</aside>

将频率最高的一个 $pair$ （若并列则取最高字典序）分配新的 $token ~ id$ 

比如上述 `word` 则新分配 `279` → `wo`

**截止状态：**

最终整个序列已经分配了 $token ~ id$（几乎不可能） 或 $token ~ id$ 的数目 = $vocab\_size$

$token ~ id \leq vocab\_size = 10^4$

---

#### 输入

二维字符串数组

#### 输出

$vocab$ : 从 $token ~ id$ 到 $bytes$ 的映射

$merges$ : 产生的 合并 $pair$（按创建顺序排列）


```python
import os, json
import regex as re
from collections import Counter
from typing import List, Dict, Tuple
```


```python
def pretokenize(text: str, pattern, special_pattern, special_lookup):
    tokens = []
    if special_pattern is not None: # 处理 <|endoftxt|>
        parts = re.split(special_pattern, text)
        matches = re.findall(special_pattern, text)
        for i, part in enumerate(parts):
            if part:
                tokens.extend([m.group(0) for m in re.finditer(pattern, part)])
            if i < len(matches):
                tokens.append(matches[i])
    else:
        tokens = [m.group(0) for m in re.finditer(pattern, text)]

    # 转成字节序列表 bytes([b]) 生成单个字节对象 | bytes(b) 生成整个字节序列
    corpus = [[bytes([b]) for b in token.encode("utf-8")] for token in tokens]
    return corpus, len(corpus)

def count_pairs(corpus, special_set):
    pairs = Counter()
    for word in corpus: # 词内合并
        for a, b in zip(word, word[1:]): # 两两配对
            if a not in special_set and b not in special_set:
                pairs[(a, b)] += 1
    return pairs

def decrement_pair(pair_counts, pair): # 删除操作，防止内存爆掉
    if pair_counts[pair] > 1:
        pair_counts[pair] -= 1
    else:
        del pair_counts[pair] # 释放 counts[b'x']为 0 的空间，提高排序效率 | 减少内存堆积

def apply_merge(corpus, pair_counts, merge_pair, special_set):
    a, b = merge_pair
    merged = a + b
    for word in corpus: # 寻找连续 a-b pair 对，替换成 merged
        i = 0
        while i < len(word) - 1: # 滑动窗口
            if word[i] == a and word[i + 1] == b:
                left = word[i - 1] if i > 0 else None
                right = word[i + 2] if i + 2 < len(word) else None
                if left and left not in special_set:
                    decrement_pair(pair_counts, (left, a))
                    pair_counts[(left, merged)] += 1
                if right and right not in special_set:
                    decrement_pair(pair_counts, (b, right))
                    pair_counts[(merged, right)] += 1
                word[i : i + 2] = [merged] # 把两个元素替换为一个
            i += 1
    if merge_pair in pair_counts: # merge_pair = [a, b]
        del pair_counts[merge_pair] # 释放内存
```


```python
def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: List[str],
    output_dir: str = "./bpeModel",
    PAT: str = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
):
    os.makedirs(output_dir, exist_ok=True)
    pattern = re.compile(PAT, re.UNICODE)

    vocab = {i: bytes([i]) for i in range(256)} # 初始化 vocab
    for tok in special_tokens:
        vocab[len(vocab)] = tok.encode("utf-8") # len(vocab) 作下标添加 special token

    special_lookup = set(special_tokens) # str 匹配
    special_set = set(tok.encode("utf-8") for tok in special_tokens) # bytes 匹配
    special_pattern = ( # 正则匹配
        re.compile("|".join(re.escape(t) for t in sorted(special_tokens, key=len, reverse=True)))
        if special_tokens else None
    )

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    corpus, n_stories = pretokenize(text, pattern, special_pattern, special_lookup)
    print(f"Loaded {n_stories} tokens")

    pair_counts = count_pairs(corpus, special_set)
    merges = []
    print(f"Initial unique pairs: {len(pair_counts)}")

    while len(vocab) < vocab_size and pair_counts:
        # item[0]: (a, b)  item[1]: freq
        (a, b), freq = max(pair_counts.items(), key=lambda item: (item[1], item[0])) 
        vocab[len(vocab)] = a + b # 分配新 token
        merges.append((a, b))
        apply_merge(corpus, pair_counts, (a, b), special_set)
        if len(merges) % 100 == 0:
            print(f"Step {len(merges)}, merged {a+b} freq={freq}")

    vocab_out = {k: v.decode("utf-8", errors="ignore") for k, v in vocab.items()}
    with open(os.path.join(output_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab_out, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "merges.txt"), "w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(f"{a.decode('utf-8', 'ignore')} {b.decode('utf-8', 'ignore')}\n")

    print(f"BPE Training done: {len(vocab)} tokens, {len(merges)} merges.")
    return vocab, merges
```


```python
vocab, merges = train_bpe(
    input_path="./datasets/TinyStories/valid.txt",
    vocab_size=10000,
    special_tokens=["<|endoftext|>"],
    output_dir="./bpeModel"
)
```

    Loaded 4554143 tokens
    Initial unique pairs: 1048
    Step 100, merged b'ad' freq=28155
    Step 200, merged b'fu' freq=12244
    Step 300, merged b'ited' freq=6999
    Step 400, merged b' outside' freq=4795
    Step 500, merged b' sorry' freq=3499
    Step 600, merged b'lly' freq=2692
    Step 700, merged b'xt' freq=2078
    Step 800, merged b' give' freq=1728
    Step 900, merged b' curi' freq=1453
    Step 1000, merged b' having' freq=1232
    Step 1100, merged b' swim' freq=1063
    Step 1200, merged b' filled' freq=929
    Step 1300, merged b'ons' freq=792
    Step 1400, merged b' treasure' freq=691
    Step 1500, merged b'fort' freq=619
    Step 1600, merged b' plac' freq=542
    Step 1700, merged b' Wh' freq=490
    Step 1800, merged b' Mittens' freq=441
    Step 1900, merged b' comfort' freq=397
    Step 2000, merged b' which' freq=357
    Step 2100, merged b' teeth' freq=328
    Step 2200, merged b' meant' freq=298
    Step 2300, merged b' sold' freq=272
    Step 2400, merged b' lit' freq=251
    Step 2500, merged b'iff' freq=231
    Step 2600, merged b' become' freq=218
    Step 2700, merged b' bags' freq=203
    Step 2800, merged b' beak' freq=192
    Step 2900, merged b' knows' freq=182
    Step 3000, merged b' falls' freq=174
    Step 3100, merged b' sadly' freq=165
    Step 3200, merged b' yet' freq=155
    Step 3300, merged b'ric' freq=148
    Step 3400, merged b' intellig' freq=141
    Step 3500, merged b' roof' freq=134
    Step 3600, merged b' pushing' freq=128
    Step 3700, merged b' dishes' freq=123
    Step 3800, merged b'ses' freq=115
    Step 3900, merged b' leaned' freq=110
    Step 4000, merged b'seum' freq=104
    Step 4100, merged b' greedy' freq=99
    Step 4200, merged b'aggie' freq=94
    Step 4300, merged b' flies' freq=90
    Step 4400, merged b' sque' freq=85
    Step 4500, merged b' hurting' freq=81
    Step 4600, merged b' happens' freq=77
    Step 4700, merged b' boot' freq=74
    Step 4800, merged b' langu' freq=70
    Step 4900, merged b' challenge' freq=66
    Step 5000, merged b' lemonade' freq=62
    Step 5100, merged b' language' freq=58
    Step 5200, merged b' Lena' freq=55
    Step 5300, merged b' cheering' freq=52
    Step 5400, merged b' onions' freq=48
    Step 5500, merged b'expected' freq=45
    Step 5600, merged b'iggy' freq=42
    Step 5700, merged b' towels' freq=40
    Step 5800, merged b' rainbows' freq=38
    Step 5900, merged b' fairies' freq=36
    Step 6000, merged b' awake' freq=34
    Step 6100, merged b' accomplishment' freq=32
    Step 6200, merged b' Amelia' freq=30
    Step 6300, merged b' det' freq=28
    Step 6400, merged b' Everybody' freq=27
    Step 6500, merged b' siblings' freq=25
    Step 6600, merged b' lighter' freq=24
    Step 6700, merged b' displayed' freq=23
    Step 6800, merged b'tuce' freq=21
    Step 6900, merged b'ppa' freq=20
    Step 7000, merged b' Robbie' freq=20
    Step 7100, merged b' gracefully' freq=19
    Step 7200, merged b' knights' freq=18
    Step 7300, merged b' postman' freq=17
    Step 7400, merged b'Nina' freq=16
    Step 7500, merged b' Fiona' freq=16
    Step 7600, merged b' merma' freq=15
    Step 7700, merged b'filled' freq=14
    Step 7800, merged b' heaven' freq=14
    Step 7900, merged b'ippy' freq=13
    Step 8000, merged b' engines' freq=13
    Step 8100, merged b'Always' freq=12
    Step 8200, merged b' crowns' freq=12
    Step 8300, merged b'His' freq=11
    Step 8400, merged b' dump' freq=11
    Step 8500, merged b'ormation' freq=10
    Step 8600, merged b' movements' freq=10
    Step 8700, merged b' Tigers' freq=10
    Step 8800, merged b'Try' freq=9
    Step 8900, merged b' man\xc3\xa2' freq=9
    Step 9000, merged b' Swim' freq=9
    Step 9100, merged b'lo' freq=8
    Step 9200, merged b' rotted' freq=8
    Step 9300, merged b' disagreement' freq=8
    Step 9400, merged b' Greg' freq=8
    Step 9500, merged b'fived' freq=7
    Step 9600, merged b' stiff' freq=7
    Step 9700, merged b' hoot' freq=7
    BPE Training done: 10000 tokens, 9743 merges.

