#Import the required modules:
from LecturaSenalesSimuladas import *
from Modelo import *
from Training import *
from Testing import *

#Create a custom Dataset with the simulated signals from the Data directory
dataset = ApneaDataset('.\data\ApneaDetection_SimulatedSignals\SenalesCONapnea.csv', '.\data\ApneaDetection_SimulatedSignals\SenalesSINapnea.csv')

#Get signal length
input_size = dataset.signal_len()
#Split the Dataset into train, validation and test Subsets
dataset.split_dataset(train_perc = 0.6, 
                      val_perc = 0.2, 
                      test_perc = 0.2)
#Data statistics
analisis_datos = dataset.analisis_datos()
print(analisis_datos)

nombre = 'nombre_modelo' #CHANGE, name used to save the model and figures

#Create and train new model
model = Model(input_size, nombre)
model_arch = model.get_architecture()
trainer = Trainer(
    model = model, 
    trainset = dataset.trainset, 
    valset = dataset.valset, 
    n_epochs = 100, 
    batch_size = 32, 
    loss_fn = 'BCE', #NOTE: in this first version, only 'BCE' loss function is available
    optimizer = 'SGD', #NOTE: in this first version, only 'SGD' optimizer is available
    lr = 0.01, 
    momentum = 0.5, 
    text = analisis_datos + model_arch)
trainer.train(verbose = True, plot = True)
#Test new model
tester = Tester(model = model, 
                testset = dataset.testset, 
                batch_size = 32)
tester.evaluate(plot = True)
#Save new model
model.save_model()