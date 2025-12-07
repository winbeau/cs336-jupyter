## 在 huggingface 上下载数据


```python
import os
from datasets import load_dataset 

dataset = load_dataset("roneneldan/TinyStories")

print(dataset)
```


```python
os.makedirs("/home/winbeau/Study/Assign1-cs336-Jupyter/datasets/TinyStories", exist_ok=True)

# dataset['train'].to_parquet("/home/winbeau/Study/cs336-Assign1-Jupyter/datasets/TinyStories/train.parquet")
# dataset['validation'].to_parquet("/home/winbeau/Study/cs336-Assign1-Jupyter/datasets/TinyStories/valid.parquet")
```


```python
os.makedirs("/home/winbeau/Study/Assign1-cs336-Jupyter/datasets/TinyStories", exist_ok=True)

with open("/home/winbeau/Study/Assign1-cs336-Jupyter/datasets/TinyStories/train.txt", "w", encoding="utf-8") as f:
    for row in dataset['train']:
        f.write(row["text"].replace("\n", " ") + "\n")
with open("/home/winbeau/Study/Assign1-cs336-Jupyter/datasets/TinyStories/valid.txt", "w", encoding="utf-8") as f:
    for row in dataset['validation']:
        f.write(row["text"].replace("\n", " ") + "\n")
```

## 为训练数据添加 `<|endoftext|>`
- 将训练集中每个故事添加结束标志，防止故事间跨越，学习全局超长语料
- 依然是全局BPE，但是故事间有 `<|endoftext|>` 作为挡板
- **不是** 每个故事单独训练一个BPE词表再汇总\[这可能会导致出现不同的token编号，模型无法同一使用\]，而是全局统计


```python
train_raw_path = "/home/winbeau/Study/Assign1-cs336-Jupyter/datasets/TinyStories/train.txt"
valid_raw_path = "/home/winbeau/Study/Assign1-cs336-Jupyter/datasets/TinyStories/valid.txt"

# 查看前几行
def preview_txt(cnt_line, txt_path=train_raw_path): 
    with open(txt_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f): # 行号 和 内容
            if i >= cnt_line:
                break
            print(line.strip())
```


```python
preview_txt(2)
```

    One day, a little girl named Lily found a needle in her room. She knew it was difficult to play with it because it was sharp. Lily wanted to share the needle with her mom, so she could sew a button on her shirt.  Lily went to her mom and said, "Mom, I found this needle. Can you share it with me and sew my shirt?" Her mom smiled and said, "Yes, Lily, we can share the needle and fix your shirt."  Together, they shared the needle and sewed the button on Lily's shirt. It was not difficult for them because they were sharing and helping each other. After they finished, Lily thanked her mom for sharing the needle and fixing her shirt. They both felt happy because they had shared and worked together.
    Once upon a time, there was a little car named Beep. Beep loved to go fast and play in the sun. Beep was a healthy car because he always had good fuel. Good fuel made Beep happy and strong.  One day, Beep was driving in the park when he saw a big tree. The tree had many leaves that were falling. Beep liked how the leaves fall and wanted to play with them. Beep drove under the tree and watched the leaves fall on him. He laughed and beeped his horn.  Beep played with the falling leaves all day. When it was time to go home, Beep knew he needed more fuel. He went to the fuel place and got more healthy fuel. Now, Beep was ready to go fast and play again the next day. And Beep lived happily ever after.



```python
import os 

base_dir = "/home/winbeau/Study/Assign1-cs336-Jupyter/datasets/TinyStories/"

files = [
    ("train.txt", "train_with_eot.txt"), 
    ("valid.txt", "valid_with_eot.txt"),
]
```


```python
def add_endoftext(infile: str, outfile: str): 
    """行末添加 <|endoftext|> 忽略空行 """
    cnt_in, cnt_out = 0, 0
    with open(infile, "r", encoding="utf-8") as fin, open(outfile, "w", encoding="utf=8") as fout: 
        for line in fin: 
            text = line.strip()
            cnt_in += 1
            if text: 
                fout.write(text + "<|endoftext|>\n")
                cnt_out += 1
    print(f"Complete! {cnt_out} / {cnt_in}")
```


```python
for fin, fout in files: 
    add_endoftext(
        os.path.join(base_dir, fin),
        os.path.join(base_dir, fout) 
    )
print("All files have added <|endoftext|> and saved!")
```


```python
preview_txt(2, os.path.join(base_dir, "train_with_eot.txt")) # 验证是否加入成功
```

    One day, a little girl named Lily found a needle in her room. She knew it was difficult to play with it because it was sharp. Lily wanted to share the needle with her mom, so she could sew a button on her shirt.  Lily went to her mom and said, "Mom, I found this needle. Can you share it with me and sew my shirt?" Her mom smiled and said, "Yes, Lily, we can share the needle and fix your shirt."  Together, they shared the needle and sewed the button on Lily's shirt. It was not difficult for them because they were sharing and helping each other. After they finished, Lily thanked her mom for sharing the needle and fixing her shirt. They both felt happy because they had shared and worked together.<|endoftext|>
    Once upon a time, there was a little car named Beep. Beep loved to go fast and play in the sun. Beep was a healthy car because he always had good fuel. Good fuel made Beep happy and strong.  One day, Beep was driving in the park when he saw a big tree. The tree had many leaves that were falling. Beep liked how the leaves fall and wanted to play with them. Beep drove under the tree and watched the leaves fall on him. He laughed and beeped his horn.  Beep played with the falling leaves all day. When it was time to go home, Beep knew he needed more fuel. He went to the fuel place and got more healthy fuel. Now, Beep was ready to go fast and play again the next day. And Beep lived happily ever after.<|endoftext|>



```python

```
