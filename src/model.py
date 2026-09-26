import torch.nn as nn
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 2, dropout: float = 0.3):
        #super() is the parent class
        super().__init__()
        #conv_block = convolutional blocks
        #channels as in RGB channels
        #applied on every layer
        def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
            return nn.Sequential(
                #the convolutional layer
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                #BatchNorm = batch normalization
                nn.BatchNorm2d(out_channels),
                #ReLU = Rectified Linear Unit -> makes negative numbers zero
                nn.ReLU(inplace=True),
                #the convolutional filter has kernel_size x kernel_size pixels
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                #MaxPool = max pooling
                #pooling = grouping more values into one
                nn.MaxPool2d(kernel_size=2),
            )
        self.features = nn.Sequential(
            #3 represents the number of channels
            #the first number of a block has to be the same as the second
            #from the block before
            conv_block(3, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
        )
        #classifer gets the photos and decides if it s a cat or a dog
        self.classifier = nn.Sequential(
            #the average for each map
            nn.AdaptiveAvgPool2d(1),
            #transforms a multi-dimension tensor into a list
            nn.Flatten(),
            #it reduces overfitting
            nn.Dropout(dropout),
            #Linear is the completely connected layer
            nn.Linear(256, 128),
            #inplace means that it works on the spot
            nn.ReLU(inplace=True),
            #each dropout takes place before a Linear and protects it
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
if __name__ == "__main__":
    import torch
    from constants import image_size
    model = SimpleCNN()
    #dummy input = false entry
    #the inputs are batch, channels, height, width
    dummy_input = torch.randn(4, 3, image_size, image_size)
    output = model(dummy_input)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Output shape: {tuple(output.shape)}")
    print(f"Parameters: {n_params:,}")
