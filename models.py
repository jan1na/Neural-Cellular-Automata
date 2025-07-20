import torch
import torch.nn as nn
import torch.nn.functional as F

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

class NCA2(nn.Module):
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
              [0,  0,  0],
              [1,  2,  1]]], dtype=torch.float32
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
        out = out.mean(dim=(2, 3))
        return out, rgb_steps if visualize else None


class NCA3(nn.Module):
    def __init__(self, state_dim=16, num_classes=9, num_steps=8):
        super().__init__()
        self.state_dim = state_dim
        self.num_steps = num_steps

        # Learnable perception
        self.learned_perceive = nn.Conv2d(state_dim, 128, kernel_size=3, padding=1)

        # Fixed Sobel + Identity
        self.fixed_perceive = nn.Conv2d(
            state_dim, 3 * state_dim, kernel_size=3, padding=1, groups=state_dim, bias=False
        )
        self._init_fixed_filters()

        # Combined perception dimension: learned + fixed
        self.update = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(128 + 3 * state_dim, state_dim, kernel_size=1)
        )

        self.readout = nn.Sequential(
            nn.Conv2d(state_dim, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

    def _init_fixed_filters(self):
        """Initialize Sobel X, Sobel Y, and Identity filters for each channel."""
        sobel_x = torch.tensor([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]], dtype=torch.float32)
        sobel_y = torch.tensor([[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]], dtype=torch.float32)
        identity = torch.zeros((3, 3), dtype=torch.float32)
        identity[1, 1] = 1.0

        kernels = torch.stack([sobel_x, sobel_y, identity], dim=0)  # (3, 3, 3)

        kernels = kernels.unsqueeze(1)  # (3, 1, 3, 3)

        kernels = kernels.repeat(self.state_dim, 1, 1, 1)  # (3 * state_dim, 1, 3, 3)

        self.fixed_perceive.weight.data = kernels
        self.fixed_perceive.weight.requires_grad = False

    def forward(self, x, visualize=False):
        B, C, H, W = x.shape
        state = torch.zeros(B, self.state_dim, H, W, device=x.device)
        state[:, :C] = x

        rgb_steps = [x]

        for _ in range(self.num_steps):
            p_learned = self.learned_perceive(state)
            p_fixed = self.fixed_perceive(state)
            perception_vector = torch.cat([p_learned, p_fixed], dim=1)

            dx = self.update(perception_vector)
            state = state + dx

            if visualize:
                rgb_steps.append(state[:, :C].clone())

        out = self.readout(state)
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