#Import the required modules:
from DataFormatting import ApneaDataset
from Modelo import *
from Training import *
from Testing import *

#Create a custom Dataset with the simulated signals from the Data directory
X,y = ApneaDataset.from_csv(csv_path_con = '.\data\ApneaDetection_SimulatedSignals\SenalesCONapnea.csv', csv_path_sin = '.\data\ApneaDetection_SimulatedSignals\SenalesSINapnea.csv')
dataset = ApneaDataset(X,y)

#Get signal length
input_size = dataset.signal_len()
#Split the Dataset into train, validation and test Subsets
dataset.split_dataset(train_perc = 0.6, 
                      val_perc = 0.2, 
                      test_perc = 0.2)
#Data statistics
analisis_datos = dataset.analisis_datos()
print(analisis_datos)

nombre = 'modelo_prueba' #CHANGE, name used to upload the model

#Load and test model
model = Model.load_model(nombre, input_size, extension = '.pth')
print(model.get_architecture())
tester = Tester(model = model, 
                testset = dataset.testset, 
                batch_size = 32)
tester.evaluate(plot = True)