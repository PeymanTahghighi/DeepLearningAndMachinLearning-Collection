import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, 
                 in_features, 
                 out_features, 
                 kernel_size, 
                 stride=1,
                 padding = None) -> None:
        super().__init__();
        self.conv = nn.Sequential(
            nn.Conv2d(in_features, out_features, kernel_size,stride= stride, padding = kernel_size//2 if padding == None else padding, bias = False),
            nn.BatchNorm2d(out_features),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.conv(x);

class ResBlock(nn.Module, ):
    def __init__(self, 
                 in_features, 
                 out_features,
                 kernel_size = 3, 
                 stride = 1) -> None:
        super().__init__();
        self.conv1 = ConvBlock(in_features, out_features, kernel_size=1, stride=stride, padding = 0);
        self.conv2 = ConvBlock(out_features, out_features , kernel_size=3, stride=1);
        self.conv3 = nn.Conv2d(out_features, out_features*4, kernel_size=1, stride = 1, padding = 0);
        self.batch_norm = nn.BatchNorm2d(out_features*4);
        if in_features != out_features*4 or stride != 1:
            self.conv4 = ConvBlock(in_features, out_features*4, kernel_size, stride);
        self.stride = stride;
    def forward(self, x):
        out = self.conv1(x);
        out = self.conv2(out);
        out = self.conv3(out);
        if hasattr(self, 'conv4'):
            x = self.conv4(x);
        return F.relu(x + out);

class ResNet(nn.Module):
    def __init__(self, in_features, num_classes) -> None:
        super().__init__();
        self.init_conv = nn.Conv2d(in_features, 64, 7, stride = 2, padding = 3);
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1);
        self.in_features = 64;
        

        self.first_block = self._make_layer(3, self.in_features, 64, False);
        self.second_block = self._make_layer(4, self.in_features, 128, True);
        self.third_block = self._make_layer(6, self.in_features, 256, True);
        self.forth_block = self._make_layer(3, self.in_features, 512, True);
    
        self.fc = nn.Sequential(
            nn.Linear(512*4, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, num_classes)
        )
    
    def _make_layer(self, repeat, in_features, out_features, downsample = False):
        layers = [];
        layers.append(ResBlock(in_features, out_features, stride=2 if downsample is True else 1))
        self.in_features = out_features*4;
        for i in range(repeat-1):
            layers.append(ResBlock(self.in_features, out_features))
        return nn.Sequential(*layers);

    def forward(self, x):
        out = F.relu((self.init_conv(x)));
        out = self.first_block(out);
        out = self.second_block(out);
        out = self.third_block(out);
        out = self.forth_block(out);
        out = F.adaptive_avg_pool2d(out, 1).flatten(1);
        return self.fc(out);

def test():
    rs = ResNet(1,10);
    inp = torch.randn((2,1,112,112));
    out = rs(inp);
    print(out.shape);

if __name__ == "__main__":
    test();

