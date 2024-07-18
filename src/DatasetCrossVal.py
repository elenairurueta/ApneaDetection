from DataFormatting import ApneaDataset, ApneaDataset2
from Modelo import *
from Training import Trainer
from Testing import Tester
from LecturaSenalesReales import *
from LecturaAnotaciones import *
from PlotSignals import *
from Imports import *

# archivos = [4, 43, 63, 72, 84, 95]
# # ApneaDataset2.create_datasets(archivos)

# txt_archivo = ""
# for archivo in archivos: #i in range(0,5):

#     # torch.manual_seed(0) 
#     txt_archivo = f"homepap-lab-full-1600{str(archivo).zfill(3)}\n"
#     dataset = ApneaDataset2.load_dataset(f"data\ApneaDetection_HomePAPSignals\datasets\dataset2_archivo_1600{archivo:03d}.pth")
#     # dataset.resample_segments(200) 
#     analisis_datos = dataset.analisis_datos()

#     train_subsets = [0, 1, 2, 3, 4, 5, 6, 7]
#     val_subsets = [8]
#     test_subsets = [9]
#     analisis_datos += f"\nTrain: subsets {train_subsets}\nVal: subset {val_subsets}\nTest: subset {test_subsets}\n"
#     dataset.undersample_majority_class(0.0, train_subsets + val_subsets, prop = 1)
#     analisis_datos += "\nUNDERSAMPLING\n" + dataset.analisis_datos()
#     #print(analisis_datos)

#     trainset = dataset.get_subsets(train_subsets)
#     valset = dataset.get_subsets(val_subsets)
#     testset = dataset.get_subsets(test_subsets)

#     nombre = f'modelo_prueba' #CAMBIAR!! 
#     input_size = dataset.signal_len()
#     model = Model2(input_size, nombre)
#     model_arch = model.get_architecture()
#     trainer = Trainer(
#         model = model,
#         trainset = trainset,
#         valset = valset,
#         n_epochs = 1,
#         batch_size = 64,
#         loss_fn = 'BCE',
#         optimizer = 'SGD',
#         lr = 0.001,
#         momentum = 0.5,
#         text = txt_archivo + analisis_datos + model_arch)
#     trainer.train(verbose = False, plot = False, save_best_model = False)

#     # nombre = 'modelo_archivos_16000_4_43_63_72_84_95_2hp1us' ##
#     # model = Model2.load_model(nombre, input_size) ##

#     tester = Tester(model = model,
#                     testset = testset,
#                     batch_size = 64)
#     tester.evaluate(plot = False)


#---------------- Con los archivos juntos -------------------------
for i in range(0,5):
    txt_archivo = ""
    archivos = [4, 43, 63, 72, 84, 95] #
    datasets = []
    traintestval = []
    for archivo in archivos:
        txt_archivo += f"homepap-lab-full-1600{str(archivo).zfill(3)}\n"
        ds = ApneaDataset2.load_dataset(f"data\ApneaDetection_HomePAPSignals\datasets\dataset2_archivo_1600{archivo:03d}.pth")
        if(ds._ApneaDataset2__sr != 200):
            ds.resample_segments(200)
        train_subsets = [0, 1, 2, 3, 4, 5, 6, 7]
        val_subsets = [8]
        test_subsets = [9]
        datasets.append(ds)
        traintestval += [[[0, 1, 2, 3, 4, 5, 6, 7], [8], [9]]]

    joined_dataset, train_subsets, val_subsets, test_subsets = ApneaDataset2.join_datasets(datasets, traintestval)
    analisis_datos = joined_dataset.analisis_datos()
    analisis_datos += f"\nTrain: subsets {train_subsets}\nVal: subset {val_subsets}\nTest: subset {test_subsets}\n"
    joined_dataset.undersample_majority_class(0.0, train_subsets + val_subsets, prop = 1)
    analisis_datos += "\nUNDERSAMPLING\n" + joined_dataset.analisis_datos()

    joined_trainset = joined_dataset.get_subsets(train_subsets)
    joined_valset = joined_dataset.get_subsets(val_subsets)
    joined_testset = joined_dataset.get_subsets(test_subsets)
    analisis_datos += f"\nTrain count: {len(joined_trainset)}\n\tWith apnea: {int(sum(sum((joined_trainset[:][1]).tolist(), [])))}\n\tWithout apnea: {len(joined_trainset) - int(sum(sum((joined_trainset[:][1]).tolist(), [])))}\nVal count: {len(joined_valset)}\n\tWith apnea: {int(sum(sum((joined_valset[:][1]).tolist(), [])))}\n\tWithout apnea: {len(joined_valset) - int(sum(sum((joined_valset[:][1]).tolist(), [])))}\nTest count: {len(joined_testset)}\n\tWith apnea: {int(sum(sum((joined_testset[:][1]).tolist(), [])))}\n\tWithout apnea: {len(joined_testset) - int(sum(sum((joined_testset[:][1]).tolist(), [])))}"

    nombre = f'modelo_archivos_16000_4_43_63_72_84_95_2hp1us_{i}' #CAMBIAR!! 2hp1us
    input_size = joined_dataset.signal_len()
    # model = Model2(input_size, nombre)
    model = Model2.load_model(nombre, input_size)
    # model_arch = model.get_architecture()
    # trainer = Trainer(
    #     model = model,
    #     trainset = joined_trainset,
    #     valset = joined_valset,
    #     n_epochs = 100,
    #     batch_size = 32,
    #     loss_fn = 'BCE',
    #     optimizer = 'SGD',
    #     lr = 0.01,
    #     momentum = 0,
    #     text = txt_archivo + analisis_datos + model_arch)
    # trainer.train(verbose = False, plot = False, save_best_model = True)

    tester = Tester(model = model,
                    testset = joined_testset,
                    batch_size = 32)
    tester.evaluate(plot = False)