from LecturaSenales import *
from Modelo import *
from TrainValidate import *


X, y = LecturaSenalTensor()
X = torch.tensor(X, dtype=torch.float32)
input_size = len(X[0])

X = X.unsqueeze(1)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

dataset = TensorDataset(X, y)
train_size = int(0.7 * len(dataset))  # 70% para entrenamiento
val_size = int(0.2 * len(dataset))   # 20% para validación
test_size = len(dataset) - train_size - val_size  # 10% restante para prueba
trainset, valset, testset = random_split(dataset, [train_size, val_size, test_size])

print('Cantidad de datos de entrenamiento: ', len(trainset))
print('Cantidad de datos de validacion: ', len(valset))
print('Cantidad de datos de prueba: ', len(testset))

model = Modelo(input_size)
nombre = 'modelo_INTERMEDIO'

# n_epochs = 5
# model = Train(model, nombre, trainset, valset, n_epochs)
# Test(model, nombre, testset)
# saveModel(model, nombre)

model = loadModel(nombre, input_size)
Test(model, nombre, testset)
