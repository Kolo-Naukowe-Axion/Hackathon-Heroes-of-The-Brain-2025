import torch
import torch.nn as nn

class ResNetBlock(nn.Module):
    def __init__(self, hidden_dim, dropout_rate=0.5):
        super(ResNetBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return x + self.block(x)

class ResNetMLP(nn.Module):
    def __init__(self, input_dim=230, hidden_dim=1024, num_classes=4, num_blocks=3):
        super(ResNetMLP, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.res_blocks = nn.ModuleList([
            ResNetBlock(hidden_dim) for _ in range(num_blocks)
        ])
        
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.res_blocks:
            x = block(x)
        return self.classifier(x)
