from Imports import *
    
class Model(nn.Module):
    """Neural network model class"""
    def __init__(self, input_size, name:str, n_filters_1, kernel_size_Conv1, n_filters_2, kernel_size_Conv2, n_filters_3, kernel_size_Conv3, n_filters_4, kernel_size_Conv4, n_filters_5, kernel_size_Conv5, n_filters_6, kernel_size_Conv6, dropout, maxpool):
        """
        Initializes the neural network model.

        Args:
            input_size (int): size of the input data.
            nombre (str): name of the model.
        """

        super().__init__()
        self.name = name

        # Calculate output size after convolutions and pooling
        def conv_output_size(input_size, kernel_size, padding, stride, dilation = 1):
            return int(np.floor((input_size + 2*padding - dilation*(kernel_size - 1) - 1)/stride + 1))
        
        def maxpool_output_size(input_size, kernel_size, stride, padding = 0, dilation = 1):
            return int(np.floor((input_size + 2*padding - dilation*(kernel_size - 1) - 1)/stride + 1))

        # Compute the size after each layer
        size = input_size
        # size = conv_output_size(size, kernel_size_Conv1, padding=1, stride=1) #not necessary since padding='same'
        # size = conv_output_size(size, kernel_size_Conv2, padding=1, stride=1) #not necessary since padding='same'
        size = maxpool_output_size(size, kernel_size=maxpool, stride=maxpool)
        # size = conv_output_size(size, kernel_size_Conv3, padding=1, stride=1) #not necessary since padding='same'
        # size = conv_output_size(size, kernel_size_Conv4, padding=1, stride=1) #not necessary since padding='same'
        size = maxpool_output_size(size, kernel_size=maxpool, stride=maxpool)
        # size = conv_output_size(size, kernel_size_Conv5, padding=1, stride=1) #not necessary since padding='same'
        # size = conv_output_size(size, kernel_size_Conv6, padding=1, stride=1) #not necessary since padding='same'
        size = maxpool_output_size(size, kernel_size=maxpool, stride=maxpool)

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, n_filters_1, kernel_size=kernel_size_Conv1, stride=1, padding='same'),
            nn.ReLU(), 
            nn.Conv1d(n_filters_1, n_filters_2, kernel_size=kernel_size_Conv2, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(n_filters_2),
            nn.MaxPool1d(kernel_size=maxpool, stride=maxpool),
            nn.Dropout(dropout),

            nn.Conv1d(n_filters_2, n_filters_3, kernel_size=kernel_size_Conv3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv1d(n_filters_3, n_filters_4, kernel_size=kernel_size_Conv4, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(n_filters_4),
            nn.MaxPool1d(kernel_size=maxpool, stride=maxpool),
            nn.Dropout(dropout),

            nn.Conv1d(n_filters_4, n_filters_5, kernel_size=kernel_size_Conv5, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv1d(n_filters_5, n_filters_6, kernel_size=kernel_size_Conv6, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(n_filters_6),
            nn.MaxPool1d(kernel_size=maxpool, stride=maxpool),
            nn.Dropout(dropout)
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(n_filters_6 * size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
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
    
    def get_name(self):
        """
        Args: none.

        Returns:
            - str: the name of the model.
        """
        return self.name

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
            PATH = models_path + f'/{self.name + extension}'             
        torch.save(self.state_dict(), PATH)

    @staticmethod
    def load_model(models_path, name, input_size, n_filters_1, kernel_size_Conv1, n_filters_2, kernel_size_Conv2, n_filters_3, kernel_size_Conv3, n_filters_4, kernel_size_Conv4, n_filters_5, kernel_size_Conv5, n_filters_6, kernel_size_Conv6, dropout, maxpool, extension:str = '.pth', best = False):
        """
        Loads a pre-trained model from a file.

        Args:
            - nombre (str): name of the model.
            - input_size (int): size of the input data.
            - extension (str, optional): extension of the file ('.pt' or '.pth'). Defaults to '.pth'.
            
        Returns:
            - Model: the loaded model.
        """
        model = Model(input_size, name, n_filters_1, kernel_size_Conv1, n_filters_2, kernel_size_Conv2, n_filters_3, kernel_size_Conv3, n_filters_4, kernel_size_Conv4, n_filters_5, kernel_size_Conv5, n_filters_6, kernel_size_Conv6, dropout, maxpool)
        if(best):
            PATH = models_path + f'/{name}_best{extension}' 
            model.load_state_dict(torch.load(PATH))
        else:
            PATH = models_path + f'/{name + extension}'
            model.load_state_dict(torch.load(PATH))
        model.eval()
        return model
    

def init_weights(m):
    """
    Initializes the weights of the model's layers using a truncated normal distribution for the weights and zero for biases.
    """
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)




class DualInputModel(nn.Module):
    def __init__(self, eeg_input_size, sao2_input_size, name:str, 
                 n_filters_1, kernel_size_Conv1, n_filters_2, kernel_size_Conv2,
                 n_filters_3, kernel_size_Conv3, n_filters_4, kernel_size_Conv4,
                 n_filters_5, kernel_size_Conv5, n_filters_6, kernel_size_Conv6,
                 dropout, maxpool):
        super().__init__()
        self.name = name

        # EEG branch (mantengo la arquitectura original)
        self.eeg_conv_layers = nn.Sequential(
            nn.Conv1d(1, n_filters_1, kernel_size=kernel_size_Conv1, stride=1, padding='same'),
            nn.ReLU(), 
            nn.Conv1d(n_filters_1, n_filters_2, kernel_size=kernel_size_Conv2, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(n_filters_2),
            nn.MaxPool1d(kernel_size=maxpool, stride=maxpool),
            nn.Dropout(dropout),

            nn.Conv1d(n_filters_2, n_filters_3, kernel_size=kernel_size_Conv3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv1d(n_filters_3, n_filters_4, kernel_size=kernel_size_Conv4, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(n_filters_4),
            nn.MaxPool1d(kernel_size=maxpool, stride=maxpool),
            nn.Dropout(dropout),

            nn.Conv1d(n_filters_4, n_filters_5, kernel_size=kernel_size_Conv5, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv1d(n_filters_5, n_filters_6, kernel_size=kernel_size_Conv6, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(n_filters_6),
            nn.MaxPool1d(kernel_size=maxpool, stride=maxpool),
            nn.Dropout(dropout)
        )

        # SaO2 branch
        self.sao2_conv_layers = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout),
            
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout)
        )

        # Calculamos tamaños de salida para cada rama
        def maxpool_output_size(input_size, kernel_size, stride, padding = 0, dilation = 1):
            return int(np.floor((input_size + 2*padding - dilation*(kernel_size - 1) - 1)/stride + 1))
        
        eeg_size = eeg_input_size
        eeg_size = maxpool_output_size(eeg_size, kernel_size=maxpool, stride=maxpool)
        eeg_size = maxpool_output_size(eeg_size, kernel_size=maxpool, stride=maxpool)
        eeg_size = maxpool_output_size(eeg_size, kernel_size=maxpool, stride=maxpool)
        
        sao2_size = sao2_input_size
        sao2_size = maxpool_output_size(sao2_size, kernel_size=2, stride=2)
        sao2_size = maxpool_output_size(sao2_size, kernel_size=2, stride=2)

        self.fc_layers = nn.Sequential(
            nn.Linear(n_filters_6 * eeg_size + 64 * sao2_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, eeg, sao2):
        if eeg.dim() == 2:
            eeg = eeg.unsqueeze(1) 
            sao2 = sao2.unsqueeze(1) 
        x_eeg = self.eeg_conv_layers(eeg)
        x_sao2 = self.sao2_conv_layers(sao2)
        
        batch_size = eeg.size(0)
        x_eeg = x_eeg.view(batch_size, -1)
        x_sao2 = x_sao2.view(batch_size, -1)
        
        x = torch.cat((x_eeg, x_sao2), dim=1)
        x = self.fc_layers(x)
        return x

    def get_name(self):
        return self.name

    def get_architecture(self):
        return '\n\n' + str(self)

    def save_model(self, models_path, extension:str = '.pth'):
        if os.path.exists(models_path):
            PATH = models_path + f'/{self.name + extension}'             
        torch.save(self.state_dict(), PATH)

    @staticmethod
    def load_model(models_path, name, eeg_input_size,sao2_input_size, n_filters_1, kernel_size_Conv1, n_filters_2, kernel_size_Conv2, n_filters_3, kernel_size_Conv3, n_filters_4, kernel_size_Conv4, n_filters_5, kernel_size_Conv5, n_filters_6, kernel_size_Conv6, dropout, maxpool, extension:str = '.pth', best = False):
        """
        Loads a pre-trained model from a file.

        Args:
            - nombre (str): name of the model.
            - input_size (int): size of the input data.
            - extension (str, optional): extension of the file ('.pt' or '.pth'). Defaults to '.pth'.
            
        Returns:
            - Model: the loaded model.
        """
        model = DualInputModel(eeg_input_size,sao2_input_size, name, n_filters_1, kernel_size_Conv1, n_filters_2, kernel_size_Conv2, n_filters_3, kernel_size_Conv3, n_filters_4, kernel_size_Conv4, n_filters_5, kernel_size_Conv5, n_filters_6, kernel_size_Conv6, dropout, maxpool)
        if(best):
            PATH = models_path + f'/{name}_best{extension}' 
            model.load_state_dict(torch.load(PATH))
        else:
            PATH = models_path + f'/{name + extension}'
            model.load_state_dict(torch.load(PATH))
        model.eval()
        return model