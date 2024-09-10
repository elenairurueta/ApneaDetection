from DataFormatting import ApneaDataset
from Modelo import Model
from Training import Trainer
from Testing import Tester

archivo = 4 #CHANGE: 4, 43, 63, 72, 84, 95

## If the dataset has not been created, uncomment following lines:
# path_edf = 
# path_annot = 
# ApneaDataset.create_datasets([archivo], path_edf, path_annot) 

txt_archivo = f"homepap-lab-full-1600{str(archivo).zfill(3)}\n"
models_path = './models'

dataset = ApneaDataset.load_dataset(f".\data\ApneaDetection_HomePAPSignals\datasets\dataset_archivo_1600{archivo[0]:03d}.pth")
analisis_datos = dataset.data_analysis()

dataset.undersample_majority_class(0.0)
analisis_datos += '\n UNDERSAMPLED \n' + dataset.data_analysis()

nombre = f'modelo_archivo_1600{str(archivo).zfill(3)}'
input_size = dataset.signal_len()
model = Model(input_size, nombre)
model_arch = model.get_architecture()
trainer = Trainer(
    model = model, 
    trainset = dataset.trainset, 
    valset = dataset.valset, 
    n_epochs = 100, 
    batch_size = 32, 
    loss_fn = 'BCE',
    optimizer = 'SGD',
    lr = 0.01, 
    momentum = 0, 
    text = txt_archivo + analisis_datos + model_arch)
trainer.train(models_path, verbose = False, plot = False, save_model = True, save_best_model = True)

tester = Tester(model = model, 
                testset = dataset.testset, 
                batch_size = 32)
tester.evaluate(models_path, plot = False)