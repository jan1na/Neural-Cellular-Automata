import torch
import torch.nn as nn
import torch.nn.functional as F


class NCA(nn.Module):
    def __init__(self, state_dim=16, num_classes=9, num_steps=8):
        super().__init__()
        self.state_dim = state_dim
        self.num_steps = num_steps

        self.register_buffer('sobel_x', torch.tensor(
            [[[-1, 0, 1],
              [-2, 0, 2],
              [-1, 0, 1]]], dtype=torch.float32
        ).expand(state_dim, 1, 3, 3).clone())

        self.register_buffer('sobel_y', torch.tensor(
            [[[-1, -2, -1],
              [0, 0, 0],
              [1, 2, 1]]], dtype=torch.float32
        ).expand(state_dim, 1, 3, 3).clone())

        self.register_buffer('identity', torch.tensor(
            [[[0, 0, 0],
              [0, 1, 0],
              [0, 0, 0]]], dtype=torch.float32
        ).expand(state_dim, 1, 3, 3).clone())

        # Total perception: 3 x state_dim channels
        self.update_mlp = nn.Sequential(
            nn.Linear(3 * state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, state_dim)
        )

        self.readout = nn.Sequential(
            nn.Conv2d(state_dim, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

    def perceive(self, x):
        # x shape: [B, C, H, W]
        dx = F.conv2d(x, self.sobel_x, padding=1, groups=self.state_dim)
        dy = F.conv2d(x, self.sobel_y, padding=1, groups=self.state_dim)
        id = F.conv2d(x, self.identity, padding=1, groups=self.state_dim)
        # Concatenate along channel dimension
        perception = torch.cat([dx, dy, id], dim=1)
        return perception

    def forward(self, x, visualize=False):
        B, C, H, W = x.shape
        state = torch.zeros(B, self.state_dim, H, W, device=x.device)
        state[:, :C] = x

        rgb_steps = [x]

        for _ in range(self.num_steps):
            perception = self.perceive(state)
            # Flatten perception per pixel
            perception_flat = perception.permute(0, 2, 3, 1)  # B,H,W,3*C
            delta = self.update_mlp(perception_flat)
            delta = delta.permute(0, 3, 1, 2)  # B,C,H,W
            state = state + delta
            if visualize:
                rgb_steps.append(state[:, :C].clone())

        out = self.readout(state)
        out = F.adaptive_max_pool2d(out, 1).squeeze(-1).squeeze(-1)
        return out, rgb_steps if visualize else None


class CNNBaseline(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU()
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        # Compute average of all pixel values for each channel (B, 128, H, W) -> (B, 128, 1, 1)
        x = self.pool(x)
        # Flatten the output to feed into the fully connected layer (B, 128, 1, 1) -> (B, 128)
        x = x.view(x.size(0), -1)
        return self.fc(x)
