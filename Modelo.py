
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split, default_collate
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder

from PruebaLecturaSenal import LecturaSenalTensor

def ModeloPrueba():

    X, y = LecturaSenalTensor()

    X = torch.tensor(X, dtype=torch.float32)
    X = X.unsqueeze(1)  # Add a channel dimension at index 1
    y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    
    trainset, testset = random_split(TensorDataset(X, y), [0.7, 0.3])
    
    batch_size=1
    train_loader = DataLoader(trainset, batch_size, shuffle=True, num_workers=2,pin_memory=True)
    test_loader = DataLoader(testset, batch_size, num_workers=2,pin_memory=True)


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
                nn.Linear(512, 2)
            )

        def forward(self, x):
            x = self.conv_layers(x)
            x = x.view(x.size(0), -1)  # Flatten the output of the convolutional layers
            x = self.fc_layers(x)
            return x

    '''
    model = nn.Sequential(
        nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1), # 1 canal entrada, 32 salida
        nn.ReLU(),
        nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool1d(2), #el número de canales se mantiene y la dimension espacial se reduce a la mitad - output: 64 x 3000

        nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool1d(2), # output: 128 x 1500

        nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool1d(2), # output: 256 x 750

        nn.Flatten(),
        # fully connected:
        nn.Linear(192000, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 2) #clasifica en 2 clases
    )
'''
    model = Model()
    for data, label in train_loader:
        print(data.shape)
        print(label.shape)

        out = model(data)
        _, pred = torch.max(out, dim=1)
        print(pred, label)
        break


if __name__ == '__main__':
    ModeloPrueba()