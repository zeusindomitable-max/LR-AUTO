<div align="center">

# **lr-auto**  
### **Learning Rate yang Tahu Diri**

> **No Tuning. No NaN. 10% Faster.**

</div>

---

**Author**: **Hari Tedjamantri**  
**Email**: haryganteng06@gmail.com  
**X**: [@haritedjamantri](https://x.com/haritedjamantri)

---

## **Install**

```bash
pip install lr-auto
```
## Quick Start

```bash
from lr_auto import lr_auto

optimizer = torch.optim.AdamW(model.parameters(), lr=0)

for step, batch in enumerate(dataloader):
    loss.backward()
    lr_auto(optimizer, step, total_steps=10000)
    optimizer.step()
```

## Works Best With EBC + Act-Guardian

```bash
clip_fn()
guard_fn()
lr_auto(optimizer, step, total_steps)
```

## Results

Metric       | Manual | LR-AUTO
-------------|--------|---------
Final Loss   | 2.29   | **2.22**
NaN          | 1      | **0**
Speed        | 100%   | **110%**


## Citation



@software{lr-auto-2025,
  author = {Hari Tedjamantri},
  
  title = {lr-auto: Self-Adaptive Learning Rate Scheduler},
  
  year = {2025},
  
  url = {https://github.com/ebc-clip/lr-auto}
}
