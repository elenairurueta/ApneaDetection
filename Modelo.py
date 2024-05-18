from Imports import *

def Modelo():

    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()

            self.conv_layers = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(2),

                nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(2),

                nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(2)
            )

            self.fc_layers = nn.Sequential(
                nn.Linear(256 * 750, 1024),
                nn.ReLU(),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            x = self.conv_layers(x)
            x = x.view(x.size(0), -1)  # Flatten the output of the convolutional layers
            x = self.fc_layers(x)
            return x
        
    return Model()
