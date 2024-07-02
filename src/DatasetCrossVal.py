from DataFormatting import ApneaDataset, ApneaDataset2, create_datasets
from Modelo import *
from Training import Trainer
from Testing import Tester
from LecturaSenalesReales import *
from LecturaAnotaciones import *
from PlotSignals import *
archivos = [4] #, 43, 63, 72, 84, 95
txt_archivo = ""
for archivo in archivos:
    txt_archivo += f"homepap-lab-full-1600{str(archivo).zfill(3)}\n"

for archivo in archivos:
        path_edf = f"C:\\Users\\elena\\OneDrive\\Documentos\\Tesis\\Dataset\\HomePAP\\polysomnography\\edfs\\lab\\full\\homepap-lab-full-1600{archivo:03d}.edf"
        path_annot = f"C:\\Users\\elena\\OneDrive\\Documentos\\Tesis\\Dataset\\HomePAP\\polysomnography\\annotations-events-profusion\\lab\\full\\homepap-lab-full-1600{archivo:03d}-profusion.xml"
        all_signals = read_signals_EDF(path_edf)
        annotations = Anotaciones(path_annot)

        bipolar_signal, tiempo, sampling = get_bipolar_signal(all_signals['C3'], all_signals['O1'])

        segments = get_signal_segments_strict(bipolar_signal, tiempo, sampling, annotations)

        X,y = ApneaDataset2.from_segments(segments, stand = True)
        dataset = ApneaDataset2(X,y, archivo)

        dataset.split_dataset()
        analisis_datos = dataset.analisis_datos()
        print(analisis_datos)
        dataset.save_dataset(f"data\ApneaDetection_HomePAPSignals\datasets\dataset2_archivo_1600{archivo:03d}.pth")

trainset = dataset.get_subsets([0, 1, 2, 3, 4, 5, 6, 7])
valset = dataset.subsets[8]
testset = dataset.subsets[9]

nombre = f'modelo_'
input_size = dataset.signal_len()
model = Model2(input_size, nombre)
model_arch = model.get_architecture()
trainer = Trainer(
    model = model, 
    trainset = trainset, 
    valset = valset, 
    n_epochs = 3, 
    batch_size = 32, 
    loss_fn = 'BCE', #NOTE: in this first version, only 'BCE' loss function is available
    optimizer = 'SGD', #NOTE: in this first version, only 'SGD' optimizer is available
    lr = 0.01, 
    momentum = 0, 
    text = txt_archivo + analisis_datos + model_arch)
trainer.train(verbose = True, plot = True, save_best_model = True)

tester = Tester(model = model, 
                testset = testset, 
                batch_size = 32)
tester.evaluate(plot = True)