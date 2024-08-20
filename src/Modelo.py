try:
    from Imports import *
except:
    from src.Imports import *
    
class Model1(nn.Module):
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
        x = x.view(batch_size, -1) #flatten
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

    def save_model(self, extension:str = '.pth'):
        """
        Saves the parameters of the model to a file.

        Args:
            - extension (str, optional): extension of the file ('.pt' or '.pth'). Defaults to '.pth'.

        Returns: none
        """
        if os.path.exists(f'D:/models'):
            if not os.path.exists(f'D:/models/{self.nombre}'):
                os.makedirs(f'D:/models/{self.nombre}')
            PATH = f'D:/models/{self.nombre}/{self.nombre + extension}'
        elif os.path.exists(f'./models'):
            if not os.path.exists(f'./models/{self.nombre}'):
                os.makedirs(f'./models/{self.nombre}')
            PATH = f'./models/{self.nombre}/{self.nombre + extension}'
        torch.save(self.state_dict(), PATH)

    @staticmethod
    def load_model(nombre, input_size, extension:str = '.pth'):
        """
        Loads a pre-trained model from a file.

        Args:
            - nombre (str): name of the model.
            - input_size (int): size of the input data.
            - extension (str, optional): extension of the file ('.pt' or '.pth'). Defaults to '.pth'.
            
        Returns:
            - Model: the loaded model.
        """
        model = Model1(input_size, nombre)
        try:
            PATH = f'D:/models/{nombre}/{nombre + extension}'
            model.load_state_dict(torch.load(PATH))
        except:
            PATH = f'./models/{nombre}/{nombre + extension}'
            model.load_state_dict(torch.load(PATH))
        model.eval()
        return model
    
class Model2(nn.Module):
    """Neural network model class"""
    def __init__(self, input_size, nombre:str, kernel_size_Conv1=3, kernel_size_Conv2=3, kernel_size_Conv3=3, kernel_size_Conv4=3, kernel_size_Conv5=3, kernel_size_Conv6=3, dropout=0.5):
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
            nn.Conv1d(1, 32, kernel_size=kernel_size_Conv1, stride=1, padding=1), #output (batch_size, 32, input_size)
            nn.ReLU(), 
            #nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=kernel_size_Conv2, stride=1, padding=1), #output (batch_size, 64, input_size)
            nn.ReLU(), #ReLU
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2), #output (batch_size, 64, input_size/2)
            nn.Dropout(dropout),

            nn.Conv1d(64, 128, kernel_size=kernel_size_Conv3, stride=1, padding=1), #output (batch_size, 128, input_size/2)
            nn.ReLU(), 
            #nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=kernel_size_Conv4, stride=1, padding=1), #output (batch_size, 128, input_size/2)
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(2), #output (batch_size, 128, input_size/4)
            nn.Dropout(dropout),

            nn.Conv1d(128, 256, kernel_size=kernel_size_Conv5, stride=1, padding=1), #output (batch_size, 256, input_size/4)
            nn.ReLU(),
            #nn.BatchNorm1d(256),
            nn.Conv1d(256, 256, kernel_size=kernel_size_Conv6, stride=1, padding=1), #output (batch_size, 256, input_size/4)
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.MaxPool1d(2), #output (batch_size, 256, input_size/8)
            nn.Dropout(dropout)
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
        
    def calculate_output_size(input_size, kernel_size, padding, stride):
        return (input_size - kernel_size + 2 * padding) // stride + 1

    def forward(self, x):
        """
        Defines the forward pass of the neural network model.
        """
        x = self.conv_layers(x)
        batch_size = x.size(0)
        x = x.view(batch_size, -1) #flatten
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
        # if os.path.exists(f'D:/models'):
        #     if not os.path.exists(f'D:/models/{self.nombre}'):
        #         os.makedirs(f'D:/models/{self.nombre}')
        #     PATH = f'D:/models/{self.nombre}/{self.nombre + extension}'
        # elif os.path.exists(f'./models'):
        #     if not os.path.exists(f'./models/{self.nombre}'):
        #         os.makedirs(f'./models/{self.nombre}')
        #     PATH = f'./models/{self.nombre}/{self.nombre + extension}'
        # if os.path.exists(f'/home/elena/media/disk/_cygdrive_D_models'):
        #     if not os.path.exists(f'/home/elena/media/disk/_cygdrive_D_models/{self.nombre}'):  ##CAMBIAR
        #         os.makedirs(f'/home/elena/media/disk/_cygdrive_D_models/{self.nombre}') ##CAMBIAR
        #     PATH = f'/home/elena/media/disk/_cygdrive_D_models/{self.nombre}/{self.nombre + extension}'   
        if os.path.exists(models_path):
            if not os.path.exists(models_path + f'/{self.nombre}'):
                os.makedirs(models_path + f'/{self.nombre}')
            PATH = models_path + f'/{self.nombre}/{self.nombre + extension}'             
        torch.save(self.state_dict(), PATH)

    @staticmethod
    def load_model(models_path, nombre, input_size, extension:str = '.pth', best = False):
        """
        Loads a pre-trained model from a file.

        Args:
            - nombre (str): name of the model.
            - input_size (int): size of the input data.
            - extension (str, optional): extension of the file ('.pt' or '.pth'). Defaults to '.pth'.
            
        Returns:
            - Model: the loaded model.
        """
        model = Model2(input_size, nombre)
        if(best):
            # PATH = f'/home/elena/media/disk/_cygdrive_D_models/{nombre}/{nombre}_best{extension}' 
            PATH = models_path + f'/{nombre}/{nombre}_best{extension}' 
            model.load_state_dict(torch.load(PATH))
            # except:
            #     PATH = f'./models/{nombre}/{nombre}_best{extension}'
            #     model.load_state_dict(torch.load(PATH))
        else:
            #try:
            # PATH = f'/home/elena/media/disk/_cygdrive_D_models/{nombre}/{nombre + extension}' 
            PATH = models_path + f'/{nombre}/{nombre + extension}'
            model.load_state_dict(torch.load(PATH))
            # except:
            #     PATH = f'./models/{nombre}/{nombre + extension}'
            #     model.load_state_dict(torch.load(PATH))
        model.eval()
        return model
    
class Model3(nn.Module):
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
            nn.Conv1d(1, 8, kernel_size=35, stride=1), #(input_size - (35-1) - 1)/1 + 1 = input_size - 34
            nn.ELU(),
            nn.BatchNorm1d(8),
            nn.MaxPool1d(7, stride=1), #(input_size - 34) - 7 + 1 = input_size - 40
            nn.Dropout(0.5),

            nn.Conv1d(8, 128, kernel_size=175, stride=1), #(input_size - 40) - 175 + 1 = input_size - 214
            nn.ELU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(7, stride=1), #(input_size - 214) - 7 + 1 = input_size - 220
            nn.Dropout(0.5),

            nn.Conv1d(128, 16, kernel_size=175, stride=1), #(input_size - 220) - 175 + 1 = input_size - 394
            nn.ELU(),
            nn.BatchNorm1d(16),
            nn.MaxPool1d(7, stride=1), #(input_size - 394) - 7 + 1 = input_size - 400
            nn.Dropout(0.5)
        )
        #Fully connected layers:
        self.fc_layers = nn.Sequential(
            nn.Linear(16 * (input_size - 400), 64),
            #nn.Linear(64, 1),
            nn.Softmax() 
            )
          
    def forward(self, x):
        """
        Defines the forward pass of the neural network model.
        """
        x = self.conv_layers(x)
        batch_size = x.size(0)
        x = x.view(batch_size, -1) #flatten
        nn.ELU(),
        nn.Dropout(0.5)
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

    def save_model(self, extension:str = '.pth'):
        """
        Saves the parameters of the model to a file.

        Args:
            - extension (str, optional): extension of the file ('.pt' or '.pth'). Defaults to '.pth'.

        Returns: none
        """
        if os.path.exists(f'D:/models'):
            if not os.path.exists(f'D:/models/{self.nombre}'):
                os.makedirs(f'D:/models/{self.nombre}')
            PATH = f'D:/models/{self.nombre}/{self.nombre + extension}'
        elif os.path.exists(f'./models'):
            if not os.path.exists(f'./models/{self.nombre}'):
                os.makedirs(f'./models/{self.nombre}')
            PATH = f'./models/{self.nombre}/{self.nombre + extension}'
        torch.save(self.state_dict(), PATH)

    @staticmethod
    def load_model(nombre, input_size, extension:str = '.pth'):
        """
        Loads a pre-trained model from a file.

        Args:
            - nombre (str): name of the model.
            - input_size (int): size of the input data.
            - extension (str, optional): extension of the file ('.pt' or '.pth'). Defaults to '.pth'.
            
        Returns:
            - Model: the loaded model.
        """
        model = Model3(input_size, nombre)
        try:
            PATH = f'D:/models/{nombre}/{nombre + extension}'
            model.load_state_dict(torch.load(PATH))
        except:
            PATH = f'./models/{nombre}/{nombre + extension}'
            model.load_state_dict(torch.load(PATH))
        model.eval()
        return model
    
class Model(nn.Module):
    """Neural network model class"""
    def __init__(self, input_size, nombre:str, kernel_size_Conv1=3, kernel_size_Conv2=3, kernel_size_Conv3=3, kernel_size_Conv4=3, kernel_size_Conv5=3, kernel_size_Conv6=3, dropout=0.5):
        """
        Initializes the neural network model.

        Args:
            input_size (int): size of the input data.
            nombre (str): name of the model.
        """

        super().__init__()
        self.nombre = nombre

        # Calculate output size after convolutions and pooling
        def conv_output_size(input_size, kernel_size, padding, stride, dilation = 1):
            return int(np.floor((input_size + 2*padding - dilation*(kernel_size - 1) - 1)/stride + 1))
        
        def maxpool_output_size(input_size, kernel_size, stride, padding = 0, dilation = 1):
            return int(np.floor((input_size + 2*padding - dilation*(kernel_size - 1) - 1)/stride + 1))

        # Compute the size after each layer
        size = input_size
        size = conv_output_size(size, kernel_size_Conv1, padding=1, stride=1)
        size = conv_output_size(size, kernel_size_Conv2, padding=1, stride=1)
        size = maxpool_output_size(size, kernel_size=2, stride=2)
        size = conv_output_size(size, kernel_size_Conv3, padding=1, stride=1)
        size = conv_output_size(size, kernel_size_Conv4, padding=1, stride=1)
        size = maxpool_output_size(size, kernel_size=2, stride=2)
        size = conv_output_size(size, kernel_size_Conv5, padding=1, stride=1)
        size = conv_output_size(size, kernel_size_Conv6, padding=1, stride=1)
        size = maxpool_output_size(size, kernel_size=2, stride=2)

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=kernel_size_Conv1, stride=1, padding=1),
            nn.ReLU(), 
            nn.Conv1d(32, 64, kernel_size=kernel_size_Conv2, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),

            nn.Conv1d(64, 128, kernel_size=kernel_size_Conv3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=kernel_size_Conv4, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),

            nn.Conv1d(128, 256, kernel_size=kernel_size_Conv5, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=kernel_size_Conv6, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.MaxPool1d(2),
            nn.Dropout(dropout)
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * size, 1024), # Size after conv layers
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten for the fully connected layers
        x = self.fc_layers(x)
        return x
   

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
            if not os.path.exists(models_path + f'/{self.nombre}'):
                os.makedirs(models_path + f'/{self.nombre}')
            PATH = models_path + f'/{self.nombre}/{self.nombre + extension}'             
        torch.save(self.state_dict(), PATH)

    @staticmethod
    def load_model(models_path, nombre, input_size, extension:str = '.pth', best = False):
        """
        Loads a pre-trained model from a file.

        Args:
            - nombre (str): name of the model.
            - input_size (int): size of the input data.
            - extension (str, optional): extension of the file ('.pt' or '.pth'). Defaults to '.pth'.
            
        Returns:
            - Model: the loaded model.
        """
        model = Model(input_size, nombre)
        if(best):
            PATH = models_path + f'/{nombre}/{nombre}_best{extension}' 
            model.load_state_dict(torch.load(PATH))
        else:
            PATH = models_path + f'/{nombre}/{nombre + extension}'
            model.load_state_dict(torch.load(PATH))
        model.eval()
        return model