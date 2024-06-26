#Import the required modules:
from DataFormatting import ApneaDataset
from Modelo import *
from Training import Trainer
from Testing import Tester
from LecturaSenalesReales import *
from LecturaAnotaciones import *

archivo = 63
txt_archivo = f"homepap-lab-full-1600{str(archivo).zfill(3)}\n"
path_edf = f"C:\\dev\\ApneaDetection\\data\\ApneaDetection_HomePAPSignals\\edfs\\homepap-lab-full-1600{str(archivo).zfill(3)}.edf"
path_annot = f"C:\\dev\\ApneaDetection\\data\\ApneaDetection_HomePAPSignals\\xmls\\homepap-lab-full-1600{str(archivo).zfill(3)}-profusion.xml"
all_signals = read_signals_EDF(path_edf)
annotations = Anotaciones(path_annot)


bipolar_signal, tiempo, sampling = get_bipolar_signal(all_signals['C3'], all_signals['O1'])
segments = get_signal_segments_strict(bipolar_signal, tiempo, sampling, annotations)
X,y = ApneaDataset.from_segments(segments)
dataset = ApneaDataset(X,y, archivo)

#Get signal length
input_size = dataset.signal_len()
#Split the Dataset into train, validation and test Subsets
dataset.split_dataset(train_perc = 0.6, 
                      val_perc = 0.2, 
                      test_perc = 0.2)
#Data statistics
analisis_datos = dataset.analisis_datos()
print(analisis_datos)
dataset.undersample_majority_class(0.0)
analisis_datos = dataset.analisis_datos()
print(analisis_datos)

nombre = 'modelo_senalesreales' #CHANGE, name used to save the model and figures

#Create and train new model
model = Model1(input_size, nombre)
model_arch = model.get_architecture()
trainer = Trainer(
    model = model, 
    trainset = dataset.trainset, 
    valset = dataset.valset, 
    n_epochs = 2, 
    batch_size = 8, 
    loss_fn = 'BCE', #NOTE: in this first version, only 'BCE' loss function is available
    optimizer = 'SGD', #NOTE: in this first version, only 'SGD' optimizer is available
    lr = 0.01, 
    momentum = 0.5, 
    text = txt_archivo + analisis_datos + model_arch)
trainer.train(verbose = True, plot = True)
#Test new model
tester = Tester(model = model, 
                testset = dataset.testset, 
                batch_size = 8)
tester.evaluate(plot = True)
#Save new model
model.save_model()