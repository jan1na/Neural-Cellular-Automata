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
            nn.Conv2d(state_dim, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

    def forward(self, x, visualize=False):
        # batch size, channels (RGB), height, width
        B, C, H, W = x.shape
        # extend channels to state_dim (often 16 in NCA literature)
        state = torch.zeros(B, self.state_dim, H, W, device=x.device)
        # set first C channels to input x, rest are zeros
        state[:, :C] = x

        rgb_steps = [x]

        # update state for num_steps
        for _ in range(self.num_steps):
            perception_vector = self.perceive(state)
            dx = self.update(perception_vector)
            state = state + dx
            if visualize:
                rgb_steps.append(state[:, :C].clone())

        out = self.readout(state)
        # mean over all class logit over all pixels
        out = out.mean(dim=(2, 3))
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