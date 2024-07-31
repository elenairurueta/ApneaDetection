import torch
from Modelo import Model2
from Training import Trainer 
from Testing import Tester
from DataFormatting import ApneaDataset2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = ApneaDataset2.load_dataset(f"..\\data\\ApneaDetection_HomePAPSignals\\datasets\\dataset2_archivo_1600004.pth")
analisis_datos = dataset.analisis_datos()

train_subsets = [0, 1, 2, 3, 4, 5, 6, 7]
val_subsets = [8]
test_subsets = [9]
analisis_datos += f"\nTrain: subsets {train_subsets}\nVal: subset {val_subsets}\nTest: subset {test_subsets}\n"
dataset.undersample_majority_class(0.0, train_subsets + val_subsets, prop = 1)
analisis_datos += "\nUNDERSAMPLING\n" + dataset.analisis_datos()

trainset = dataset.get_subsets(train_subsets)
valset = dataset.get_subsets(val_subsets)
testset = dataset.get_subsets(test_subsets)

nombre = f'modelo_prueba'
input_size = dataset.signal_len()
model = Model2(input_size, nombre).to(device)
print(next(model.parameters()).device)

trainer = Trainer(
    model = model,
    trainset = trainset,
    valset = valset,
    n_epochs = 1,
    batch_size = 32,
    loss_fn = 'BCE',
    optimizer = 'SGD',
    lr = 0.01,
    momentum = 0,
    text = '',
    device = device)
trainer.train(verbose = True, plot = False, save_best_model = True)
best_model = Model2.load_model(nombre, input_size, best = True)
tester = Tester(model = best_model,
                testset = testset,
                batch_size = 32, 
                device = device)
cm, metrics = tester.evaluate(plot = False)