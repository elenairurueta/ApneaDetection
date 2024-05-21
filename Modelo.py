from Imports import *

def Modelo(input_size):

    class Model(nn.Module):
        def __init__(self, input_size):
            super().__init__()

            self.conv_layers = nn.Sequential(

                #entrada de dimensiones (batch_size, 1, input_size)
                nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1), #output (batch_size, 32, input_size)
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1), #output (batch_size, 64, input_size)
                nn.ReLU(),
                nn.MaxPool1d(2), #output (batch_size, 64, input_size/2)

                nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1), #output (batch_size, 128, input_size/2)
                nn.ReLU(),
                nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1), #output (batch_size, 128, input_size/2)
                nn.ReLU(),
                nn.MaxPool1d(2), #output (batch_size, 128, input_size/4)

                nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1), #output (batch_size, 256, input_size/4)
                nn.ReLU(),
                nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1), #output (batch_size, 256, input_size/4)
                nn.ReLU(),
                nn.MaxPool1d(2) #output (batch_size, 256, input_size/8)
            )

            self.fc_layers = nn.Sequential(
                nn.Linear(256 * int(input_size/8), 1024), #(batch_size, 32000) -> (batch_size, 1024)
                nn.ReLU(),
                nn.Linear(1024, 512), #(batch_size, 1024) -> (batch_size, 512)
                nn.ReLU(),
                nn.Linear(512, 1), #(batch_size, 512) -> (batch_size, 1)
                nn.Sigmoid()
            )

        def forward(self, x):
            x = self.conv_layers(x)
            batch_size = x.size(0)
            x = x.view(batch_size, -1)  # Flatten the output of the convolutional layers: (batch_size, 256, 750) -> (batch_size, 192000)
            x = self.fc_layers(x)
            return x
        
    return Model(input_size)


def saveModel(model, PATH):
    torch.save(model.state_dict(), PATH)

def loadModel(PATH, input_size):
    model = Modelo(input_size)
    model.load_state_dict(torch.load(PATH))
    model.eval()
    return model
