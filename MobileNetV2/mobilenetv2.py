import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, stride = 1, seperable = False) -> None:
        super().__init__();

        if seperable is False:
            self.net = nn.Sequential(
                nn.Conv2d(in_features, out_features, kernel_size=kernel_size, stride=stride, padding=kernel_size//2),
                nn.BatchNorm2d(out_features),
                nn.LeakyReLU(0.2)
            )
        else:
            self.net = nn.Sequential(
                nn.Conv2d(in_features, out_features, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=in_features),
                nn.Conv2d(in_features, out_features, kernel_size=1, stride=1, padding=0,),
                nn.BatchNorm2d(out_features),
                nn.LeakyReLU(0.2)
            )
    def forward(self, x):
        return self.net(x);

class InvertedResBlock(nn.Module):
    def __init__(self, in_features, out_features, stride= 1, growth=6) -> None:
        super().__init__();
        self.net = nn.Sequential(ConvBlock(in_features, in_features*growth, 1, 1),
                                 ConvBlock(in_features*growth, in_features*growth, 3, stride, seperable=True),
                                 nn.Conv2d(in_features*growth, out_features, 1, 1));
        self.conv = nn.Conv2d(in_features, out_features, 3, padding=1, stride=stride);
    
    def forward(self, x):
        out = self.net(x);
        x = self.conv(x);

        return out+x;





    

class MobileNetV2(nn.Module):
    def __init__(self, in_channels, num_classes, alpha = 1) -> None:
        super().__init__();

        self.init_conv = nn.Conv2d(in_channels, 32, 3, 2, 1);
        layers = [];

        layers.append(InvertedResBlock(32, 16, 1, 1));
        layers.append(InvertedResBlock(16, 24, 2));
        layers.append(InvertedResBlock(24, 32, 2));
        layers.append(InvertedResBlock(32, 64, 2));
        layers.append(InvertedResBlock(64, 96, 1));
        layers.append(InvertedResBlock(96, 160, 2));
        layers.append(InvertedResBlock(160, 320, 1));
        layers.append(ConvBlock(320,1280,1,1));
        layers.append(nn.Sequential(nn.AvgPool2d(7),
            nn.Conv2d(1280, num_classes, 1, 1, 0)));
        
        self.net = nn.Sequential(*layers);

    def forward(self, x):
        x = self.init_conv(x);
        out = self.net(x);
        return out.flatten(1);

def test():
    t = torch.randn((4,3,224,224));
    model = MobileNetV2(3,10,1);
    out = model(t);
    print(out.shape);

if __name__ == "__main__":
    test();

