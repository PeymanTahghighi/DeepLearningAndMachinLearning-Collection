import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """Convolutional block with optional depthwise separable convolution."""
    def __init__(self, in_features, out_features, kernel_size, stride=1, separable=False) -> None:
        super().__init__()
        if not separable:
            self.net = nn.Sequential(
                nn.Conv2d(in_features, out_features, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2),
                nn.BatchNorm2d(out_features),
                nn.LeakyReLU(0.2)
            )
        else:
            self.net = nn.Sequential(
                nn.Conv2d(in_features, out_features, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=in_features),
                nn.Conv2d(in_features, out_features, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(out_features),
                nn.LeakyReLU(0.2)
            )

    def forward(self, x):
        return self.net(x)

class MobileNet(nn.Module):
    """MobileNet model with depthwise separable convolutions."""
    def __init__(self, in_channels, num_classes, alpha=1) -> None:
        super().__init__()
        self.init_conv = nn.Conv2d(in_channels, 32, 3, 2, 1)
        layers = []

        layers.append(ConvBlock(32, 32, 3, 1, separable=True))
        layers.append(ConvBlock(32, 64, 1, stride=1))
        layers.append(ConvBlock(64, 64, 3, stride=2, separable=True))
        layers.append(ConvBlock(64, 128, 1, stride=1))
        layers.append(ConvBlock(128, 128, 3, stride=1, separable=True))
        layers.append(ConvBlock(128, 128, 1, stride=1))
        layers.append(ConvBlock(128, 128, 3, stride=2, separable=True))
        layers.append(ConvBlock(128, 256, 1, stride=1))
        layers.append(ConvBlock(256, 256, 3, stride=1, separable=True))
        layers.append(ConvBlock(256, 256, 1, stride=1))
        layers.append(ConvBlock(256, 256, 3, stride=2, separable=True))
        layers.append(ConvBlock(256, 512, 1, stride=1))
        
        for _ in range(5):
            layers.append(ConvBlock(512, 512, 3, stride=1, separable=True))
            layers.append(ConvBlock(512, 512, 1, stride=1))
        
        layers.append(ConvBlock(512, 512, 3, stride=2, separable=True))
        layers.append(ConvBlock(512, 1024, 1, stride=1))
        layers.append(ConvBlock(1024, 1024, 1, stride=2, separable=True))
        layers.append(nn.AdaptiveAvgPool2d((1)))
        
        self.fc = nn.Sequential(
            nn.Linear(1024, 1000),
            nn.GELU(),
            nn.Linear(1000, num_classes)
        )
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.init_conv(x)
        out = self.net(x)
        out = out.flatten(1)
        out = self.fc(out)
        return out

def test():
    """Test function for MobileNet."""
    t = torch.randn((4, 3, 224, 244))
    model = MobileNet(3, 10, 1)
    out = model(t)
    print(out.shape)

if __name__ == "__main__":
    test()
