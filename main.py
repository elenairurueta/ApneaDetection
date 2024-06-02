from LecturaSenalesSimuladas import *
from Modelo import *
from Training import *
from Testing import *

dataset = ApneaDataset('SenalesCONapnea.csv', 'SenalesSINapnea.csv')
input_size = dataset.signal_len()
dataset.split_dataset(train_perc = 0.6, 
                      val_perc = 0.2, 
                      test_perc = 0.2)
print(dataset.analisis_datos())

nombre = 'modelo_prueba'

## Nuevo modelo ##
model = Model(input_size, nombre)
trainer = Trainer(
    model = model, 
    trainset = dataset.trainset, 
    valset = dataset.valset, 
    n_epochs = 5, 
    batch_size = 32, 
    loss_fn = 'BCE', 
    optimizer = 'SGD', 
    lr = 0.01, 
    momentum = 0.9)
trainer.train(verbose = True, plot = True)
tester = Tester(model = model, 
                testset = dataset.testset, 
                batch_size = 32)
tester.evaluate(plot = True)
model.save_model()

## Cargar modelo ##
# model = Model.load_model(nombre, input_size)
# tester = Tester(model = model, 
#                 testset = dataset.testset, 
#                 batch_size = 32)
# tester.evaluate(plot = True)