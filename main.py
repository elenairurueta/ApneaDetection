from LecturaSenales import *
from Modelo import *
from TrainValidate import *


X, y = LecturaSenalTensor()
X = torch.tensor(X, dtype=torch.float32)
input_size = len(X[0])

X = X.unsqueeze(1)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

dataset = TensorDataset(X, y)
train_size = int(0.6 * len(dataset))  # 60% para entrenamiento
val_size = int(0.2 * len(dataset))   # 20% para validación
test_size = len(dataset) - train_size - val_size  # 20% restante para prueba
trainset, valset, testset = random_split(dataset, [train_size, val_size, test_size])
text = 'Cantidad de datos de entrenamiento: ' + str(len(trainset)) + '\n' + 'Cantidad de datos de validacion: ' + str(len(valset)) + '\n' + 'Cantidad de datos de prueba: ' + str(len(testset))
print(text)

model = Modelo(input_size)
nombre = 'modelo_230524'

# n_epochs = 100
# model = Train(model, nombre, trainset, valset, n_epochs, text)
# Test(model, nombre, testset)
# saveModel(model, nombre)

model = loadModel(nombre, input_size)
Test(model, nombre, testset)
