from LecturaSenales import *
from Modelo import *
from TrainValidate import *

model = Modelo()

X, y = LecturaSenalTensor()
X = torch.tensor(X, dtype=torch.float32)
X = X.unsqueeze(1)  # Add a channel dimension at index 1
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
trainset, testset = random_split(TensorDataset(X, y), [0.7, 0.3])

Train1(model, trainset, testset)