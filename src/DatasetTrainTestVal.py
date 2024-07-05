from DataFormatting import ApneaDataset
from Modelo import *
from Training import Trainer
from Testing import Tester
from LecturaSenalesReales import *
from LecturaAnotaciones import *
from PlotSignals import *

archivos = [95] #4, 43, 63, 72, 84, 
# ApneaDataset.create_datasets(archivos)

txt_archivo = ""
for archivo in archivos:
    txt_archivo += f"homepap-lab-full-1600{str(archivo).zfill(3)}\n"

dataset = ApneaDataset.load_dataset(f"data\\ApneaDetection_HomePAPSignals\\datasets\\dataset_archivo_1600{archivos[0]:03d}.pth")
analisis_datos = dataset.analisis_datos()
print(analisis_datos)
dataset.undersample_majority_class(0.0)
analisis_datos = dataset.analisis_datos()
print(analisis_datos)

nombre = f'modelo_'
input_size = dataset.signal_len()
model = Model2(input_size, nombre)
model_arch = model.get_architecture()
trainer = Trainer(
    model = model, 
    trainset = dataset.trainset, 
    valset = dataset.valset, 
    n_epochs = 3, 
    batch_size = 32, 
    loss_fn = 'BCE', #NOTE: in this first version, only 'BCE' loss function is available
    optimizer = 'SGD', #NOTE: in this first version, only 'SGD' optimizer is available
    lr = 0.01, 
    momentum = 0, 
    text = txt_archivo + analisis_datos + model_arch)
trainer.train(verbose = False, plot = False, save_best_model = True)

tester = Tester(model = model, 
                testset = dataset.testset, 
                batch_size = 32)
tester.evaluate(plot = False)

    