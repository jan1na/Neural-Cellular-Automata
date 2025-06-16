import torch
import torch.nn as nn

class NCA(nn.Module):
    def __init__(self, state_dim=16, num_classes=9, num_steps=8):
        super().__init__()
        self.state_dim = state_dim
        self.num_steps = num_steps
        self.perceive = nn.Conv2d(state_dim, 128, kernel_size=3, padding=1)
        self.update = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(128, state_dim, kernel_size=1)
        )
        self.readout = nn.Sequential(
            nn.Conv2d(state_dim, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, 1)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        state = torch.zeros(B, self.state_dim, H, W, device=x.device)
        state[:, :C] = x

        for _ in range(self.num_steps):
            y = self.perceive(state)
            dx = self.update(y)
            state = state + dx

        out = self.readout(state)
        out = out.mean(dim=(2, 3))
        return out