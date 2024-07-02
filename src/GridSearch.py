from itertools import product
from DataFormatting import ApneaDataset
from Modelo import *
from Training import Trainer
from Testing import Tester
from LecturaSenalesReales import *
from LecturaAnotaciones import *
from PlotSignals import *

archivo = 63
txt_archivo = f"homepap-lab-full-1600{str(archivo).zfill(3)}\n"
path_edf = f"C:\\dev\\ApneaDetection\\data\\ApneaDetection_HomePAPSignals\\edfs\\homepap-lab-full-1600{archivo:03d}.edf"
path_annot = f"C:\\dev\\ApneaDetection\\data\\ApneaDetection_HomePAPSignals\\xmls\\homepap-lab-full-1600{archivo:03d}-profusion.xml"
all_signals = read_signals_EDF(path_edf)
annotations = Anotaciones(path_annot)


bipolar_signal, tiempo, sampling = get_bipolar_signal(all_signals['C3'], all_signals['O1'])
segments = get_signal_segments_strict(bipolar_signal, tiempo, sampling, annotations)
X,y = ApneaDataset.from_segments(segments, stand = True)
dataset = ApneaDataset(X,y, archivo)

input_size = dataset.signal_len()

dataset.split_dataset(train_perc = 0.8, 
                      val_perc = 0.1, 
                      test_perc = 0.1)

dataset.undersample_majority_class(0.0)
analisis_datos = dataset.analisis_datos()



param_grid = {
    'n_epochs': [50, 100], #150
    'lr': [0.01, 0.001],
    'momentum': [0, 0.5, 0.9],
    'batch_size': [4, 8, 16, 32, 64]
}

param_combinations = list(product(param_grid['n_epochs'], param_grid['lr'], param_grid['momentum'], param_grid['batch_size']))

results = []

for n_epochs, lr, momentum, batch_size in param_combinations:
    nombre = f'modelo_GS_e{n_epochs}_lr{int(lr*10000)}_m{int(momentum*10)}_bs{batch_size}'
    model = Model2(input_size, nombre)
    model_arch = model.get_architecture()
    
    trainer = Trainer(
        model = model, 
        trainset = dataset.trainset, 
        valset = dataset.valset, 
        n_epochs = n_epochs, 
        batch_size = batch_size, 
        loss_fn = 'BCE', 
        optimizer = 'SGD', 
        lr = lr, 
        momentum = momentum, 
        text = txt_archivo + analisis_datos + model_arch
    )
    trainer.train(verbose = False, plot = False)
    
    tester = Tester(model = model, 
                    testset = dataset.testset, 
                    batch_size = batch_size)
    metrics = tester.evaluate(plot = False)
    
    results.append({
        'n_epochs': n_epochs,
        'lr': lr,
        'momentum': momentum,
        'batch_size': batch_size,
        'Accuracy': metrics["Accuracy"],
        'Precision': metrics["Precision"],
        'Sensitivity': metrics["Sensitivity"],
        'Specificity': metrics["Specificity"],
        'F1': metrics["F1"]
    })

df_results = pd.DataFrame(results)

df_results.to_excel('hyperparameter_search_results.xlsx', index=False)

print(f'Mejor combinación de hiperparámetros: \n\t Mejor accuracy: {df_results.loc[df_results["Accuracy"].idxmax()]} \n\t Mejor precision: {df_results.loc[df_results["Precision"].idxmax()]} \n\t Mejor F1: {df_results.loc[df_results["F1"].idxmax()]}')