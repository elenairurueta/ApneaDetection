from Imports import *
    
class Model(nn.Module):
    """Neural network model class"""
    def __init__(self, input_size, nombre:str):
        """
        Initializes the neural network model.

        Args:
            input_size (int): size of the input data.
            nombre (str): name of the model.
        """

        super().__init__()
        self.nombre = nombre

        #Convolutional layers:
        self.conv_layers = nn.Sequential(
            #input (batch_size, 1, input_size)
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
        #Fully connected layers:
        self.fc_layers = nn.Sequential(
            #input (batch_size, 256, input_size/8)
            nn.Linear(256 * int(input_size/8), 1024), #with input_size=1000: (batch_size, 32000) -> (batch_size, 1024)
            nn.ReLU(),
            nn.Linear(1024, 512), #(batch_size, 1024) -> (batch_size, 512)
            nn.ReLU(),
            nn.Linear(512, 1), #(batch_size, 512) -> (batch_size, 1)
            nn.Sigmoid() 
            )
        
    def forward(self, x):
        """
        Defines the forward pass of the neural network model.
        """
        x = self.conv_layers(x)
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = self.fc_layers(x)
        return x
    
    def get_nombre(self):
        """
        Args: none.

        Returns:
            - str: the name of the model.
        """
        return self.nombre

    def get_architecture(self):
        """
        Args: none.

        Returns:
            - str: the architecture of the model.
        """
        return '\n\n' + str(self)

    def save_model(self, models_path, extension:str = '.pth'):
        """
        Saves the parameters of the model to a file.

        Args:
            - extension (str, optional): extension of the file ('.pt' or '.pth'). Defaults to '.pth'.

        Returns: none
        """
        if os.path.exists(models_path):
            PATH = models_path + f'/{self.nombre + extension}'
            torch.save(self.state_dict(), PATH)

    @staticmethod
    def load_model(models_path, nombre, input_size, extension:str = '.pth', best = False):
        """
        Loads a pre-trained model from a file.

        Args:
            - nombre (str): name of the model.
            - input_size (int): size of the input data.
            - models_path
            - extension (str, optional): extension of the file ('.pt' or '.pth'). Defaults to '.pth'.
            - best: if it is the best model or not
            
        Returns:
            - Model: the loaded model.
        """
        model = Model(input_size, nombre)
        if(best):
            PATH = models_path + f'/{nombre}_best{extension}' 
            model.load_state_dict(torch.load(PATH))
        else:
            PATH = models_path + f'/{nombre + extension}'
            model.load_state_dict(torch.load(PATH))
        model.eval()
        return model