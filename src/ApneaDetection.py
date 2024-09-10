from DataFormatting import ApneaDataset
from Modelo import Model
from Training import Trainer
from Testing import Tester

file = 4 #CHANGE: 4, 43, 63, 72, 84, 95

## If the dataset has not been created, uncomment following lines:
# path_annot = "" + f"\\homepap-lab-full-1600{str(file).zfill(3)}-profusion.xml" #CHANGE
# path_edf = "" + f"\\homepap-lab-full-1600{str(file).zfill(3)}.edf" #CHANGE
# ApneaDataset.create_datasets([file], path_edf, path_annot) 

file_txt = f"homepap-lab-full-1600{str(file).zfill(3)}\n"
models_path = './models'

dataset = ApneaDataset.load_dataset(f".\data\ApneaDetection_HomePAPSignals\datasets\dataset_archivo_1600{file[0]:03d}.pth")
data_analysis = dataset.data_analysis()

name = f'modelo_archivo_1600{str(file).zfill(3)}'
input_size = dataset.signal_len()
model = Model(input_size, name)
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
    text = file_txt + data_analysis + model_arch)
trainer.train(models_path, verbose = False, plot = False, save_model = True, save_best_model = True)

tester = Tester(model = model, 
                testset = dataset.testset, 
                batch_size = 32)
tester.evaluate(models_path, plot = False)