import torch
import torch.nn as nn
from lr_auto import lr_auto

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)
    def forward(self, x): return self.fc(x)

def test_lr_auto_warmup():
    model = Net()
    optimizer = torch.optim.SGD(model.parameters(), lr=0)
    
    for step in range(100):
        lr_auto(optimizer, step, total_steps=1000, warmup_steps=50)
    
    assert optimizer.param_groups[0]['lr'] > 1e-5  # warmed up
    assert optimizer.param_groups[0]['lr'] <= 3e-4  # not exceeded max

def test_lr_auto_decay():
    model = Net()
    optimizer = torch.optim.SGD(model.parameters(), lr=0)
    
    for step in range(900, 1000):
        lr_auto(optimizer, step, total_steps=1000, warmup_steps=100)
    
    assert optimizer.param_groups[0]['lr'] < 1e-4  # decayed
