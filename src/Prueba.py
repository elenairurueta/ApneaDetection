from DataFormatting import ApneaDataset2
from Modelo import *
from Training import Trainer
from Testing import Tester
from Imports import *

device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
models_path = '.\models' #'/media/elena/Externo/models'

fold = 0
nombre0 = f'modelo_CPU_Dell'
txt_archivo = ""
archivos = [4, 43, 53, 55, 63, 72, 84, 95, 105, 113, 122, 151] 
datasets = []
traintestval = []

for archivo in archivos:
    txt_archivo += f"homepap-lab-full-1600{str(archivo).zfill(3)}\n"
    ds = ApneaDataset2.load_dataset(f".\data\ApneaDetection_HomePAPSignals\datasets\dataset2_archivo_1600{archivo:03d}.pth") #..\\data\\ApneaDetection_HomePAPSignals\\datasets\\dataset2_archivo_1600{archivo:03d}.pth")
    if(ds._ApneaDataset2__sr != 200):
        ds.resample_segments(200)
    datasets.append(ds)
    
    train_idx = [(fold + i) % 10 for i in range(8)]
    val_idx = [(fold - 2 + i) % 10 for i in range(1)]
    test_idx = [(fold - 1 + i) % 10 for i in range(1)]
    traintestval.append([train_idx, val_idx, test_idx])

if not os.path.exists(models_path + '/' + nombre0): 
    os.makedirs(models_path + '/' + nombre0) 

joined_dataset, train_subsets, val_subsets, test_subsets = ApneaDataset2.join_datasets(datasets, traintestval)
analisis_datos = joined_dataset.analisis_datos()
analisis_datos += f"\nTrain: subsets {train_subsets}\nVal: subset {val_subsets}\nTest: subset {test_subsets}\n"
joined_dataset.undersample_majority_class(0.0, train_subsets + val_subsets, prop = 1)
analisis_datos += "\nUNDERSAMPLING\n" + joined_dataset.analisis_datos()

joined_trainset = joined_dataset.get_subsets(train_subsets)
joined_valset = joined_dataset.get_subsets(val_subsets)
joined_testset = joined_dataset.get_subsets(test_subsets)
analisis_datos += f"\nTrain count: {len(joined_trainset)}\n\tWith apnea: {int(sum(sum((joined_trainset[:][1]).tolist(), [])))}\n\tWithout apnea: {len(joined_trainset) - int(sum(sum((joined_trainset[:][1]).tolist(), [])))}\nVal count: {len(joined_valset)}\n\tWith apnea: {int(sum(sum((joined_valset[:][1]).tolist(), [])))}\n\tWithout apnea: {len(joined_valset) - int(sum(sum((joined_valset[:][1]).tolist(), [])))}\nTest count: {len(joined_testset)}\n\tWith apnea: {int(sum(sum((joined_testset[:][1]).tolist(), [])))}\n\tWithout apnea: {len(joined_testset) - int(sum(sum((joined_testset[:][1]).tolist(), [])))}"

for i in range(0,1): #5

    nombre = nombre0 + f'_{i}'
    print(nombre + '\n')

    input_size = joined_dataset.signal_len()
    model = Model2(input_size, nombre).to(device)
    model_arch = model.get_architecture()
    trainer = Trainer(
        model = model,
        trainset = joined_trainset,
        valset = joined_valset,
        n_epochs = 100,
        batch_size = 32,
        loss_fn = 'BCE',
        optimizer = 'SGD',
        lr = 0.01,
        momentum = 0,
        text = txt_archivo + analisis_datos + model_arch, 
        device = device)
    trainer.train(models_path + f'/{nombre0}', verbose = False, plot = False, save_best_model = False)
    

