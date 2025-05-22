import torch
import os
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, input_size, name = 'shhs_model'):
        super(CNNModel, self).__init__()

        # three convolutional layers: convolutions had a stride size of one and used zero padding.
        # batch normalisation (after each convolution layer) and dropout (after each pooling layer)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=35, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(8)
        self.pool1 = nn.MaxPool1d(kernel_size=7, stride=1)
        self.dropout1 = nn.Dropout(p=0.1)

        self.conv2 = nn.Conv1d(in_channels=8, out_channels=128, kernel_size=175, stride=1, padding=0)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=7, stride=1)
        self.dropout2 = nn.Dropout(p=0.1)

        self.conv3 = nn.Conv1d(in_channels=128, out_channels=16, kernel_size=175, stride=1, padding=0)
        self.bn3 = nn.BatchNorm1d(16)
        self.pool3 = nn.MaxPool1d(kernel_size=7, stride=1)
        self.dropout3 = nn.Dropout(p=0.1)

        # Calculate the size of the flattened output
        conv_output_size = self._get_conv_output_size(input_size)

        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 64)
        self.dropout_fc = nn.Dropout(p=0.0)
        self.fc2 = nn.Linear(64, 2)
        self._initialize_weights()

        self.name = name

    def _get_conv_output_size(self, input_size):
            """
            Helper function to calculate the size of the output after the convolutional layers.
            """
            x = torch.zeros(1, 1, input_size)  # Create a dummy input
            x = self.pool1(F.elu(self.bn1(self.conv1(x))))
            x = self.pool2(F.elu(self.bn2(self.conv2(x))))
            x = self.pool3(F.elu(self.bn3(self.conv3(x))))
            return x.view(1, -1).size(1)  # Flatten and return the size

    def _initialize_weights(self):
    # initialized the weights of our CNN by drawing from a truncated normal distribution with zero mean
    # initialized the weights of our CNN by drawing from a truncated normal distribution with zero mean
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self,x): # each convolutional layer involved normalization, ELU activation, and max-pooling
        # each convolutional layer involved normalization, ELU activation, and max-pooling
        x = self.pool1(F.elu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)
        x = self.pool2(F.elu(self.bn2(self.conv2(x))))
        x = self.dropout2(x)
        x = self.pool3(F.elu(self.bn3(self.conv3(x))))
        x = self.dropout3(x)
        
        # the dense layer was preceded by a flattening operation and followed by ELU activation and a dropout layer
        x = x.view(x.size(0), -1) # flattening
        x = F.elu(self.fc1(x)) # dense layer with ELU activation
        x = self.dropout_fc(x) # dropout after dense layer
        x = self.fc2(x) # output layer with softmax classifier
        return F.log_softmax(x, dim=1) # the output layer used a softmax classifier

    def save_model(self, path):
        """
        Saves the CNNModel to the specified path.

        Args:
            model (CNNModel): The model to save.
            path (str): The file path where the model will be saved.
        """
        torch.save(self.state_dict(), os.path.join(path, self.name + '.pth'))
        print(f"Model saved to {path}")
