from DataFormatting import ApneaDataset, ApneaDataset2
from Modelo import *
from Training import Trainer
from Testing import Tester
from LecturaSenalesReales import *
from LecturaAnotaciones import *
from PlotSignals import *

archivos = [4, 43, 63, 72, 84, 95] #
datasets = []

#ApneaDataset2.create_datasets(archivos)

traintestval = []

for archivo in archivos:
    ds = ApneaDataset2.load_dataset(f"data\ApneaDetection_HomePAPSignals\datasets\dataset2_archivo_1600{archivo:03d}.pth")

    if(ds._ApneaDataset2__sr != 200):
        ds.resample_segments(200)

    train_subsets = [0, 1, 2, 3, 4, 5, 6, 7]
    val_subsets = [8]
    test_subsets = [9]

    datasets.append(ds)
    traintestval += [[[0, 1, 2, 3, 4, 5, 6, 7], [8], [9]]]


dataset, train_subsets, val_subsets, test_subsets = ApneaDataset2.join_datasets(datasets, traintestval)
print(dataset.analisis_datos())


