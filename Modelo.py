from Imports import *

class Model(nn.Module):

    def __init__(self, input_size, nombre:str):
        super().__init__()
        self.nombre = nombre
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
            
            # nn.Linear(256 * int(input_size/8), 2048), #(batch_size, 32000) -> (batch_size, 2048)
            # nn.ReLU(),
            # nn.Linear(2048, 1024), #(batch_size, 2048) -> (batch_size, 1024)
            # nn.ReLU(),
            # nn.Linear(1024, 512), #(batch_size, 1024) -> (batch_size, 512)
            # nn.ReLU(),
            # nn.Linear(512, 128), #(batch_size, 512) -> (batch_size, 128)
            # nn.ReLU(),
            # nn.Linear(128, 1), #(batch_size, 128) -> (batch_size, 1)
            # nn.Sigmoid()
            
            ## VIEJO: ##
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
    
    def get_nombre(self):
        return self.nombre
        
    def save_model(self, extension:str = '.pth'):
        PATH = 'models/' + self.nombre + extension
        torch.save(self.state_dict(), PATH)

    @staticmethod
    def load_model(nombre, input_size, extension:str = '.pth'):
        PATH = 'models/' + nombre + extension
        model = Model(input_size, nombre)
        model.load_state_dict(torch.load(PATH))
        model.eval()
        return model