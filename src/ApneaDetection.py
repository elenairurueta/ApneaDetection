from Imports import *
from DataFormatting import ApneaDataset
from Models import Model, init_weights
from Training import Trainer
from Testing import Tester
from Data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
files = [4, 43, 53, 55, 63, 72, 84, 95, 105, 113, 122, 151]
models_path = './models'

# path_annot = ""  #CHANGE
# path_edf = "" #CHANGE
# ApneaDataset.create_datasets(files, path_edf, path_annot) 

txt_file = ""
name0 = f'model_file_1600' + "_".join([str(f).zfill(3) for f in files])

for fold in range(0,9):        
    name1 = name0 + f'_fold{fold}'
    datasets = []
    traintestval = []

    for file in files: 
        txt_file += f"homepap-lab-full-1600{str(file).zfill(3)}\n"
        ds = ApneaDataset.load_dataset(f"./data/ApneaDetection_HomePAPSignals/datasets/dataset2_archivo_1600{file:03d}.pth")
        if(ds._ApneaDataset__sr != 200):
            ds.resample_segments(200)
        datasets.append(ds)
        
        train_idx = [(fold + i) % 10 for i in range(8)]
        val_idx = [(fold - 2 + i) % 10 for i in range(1)]
        test_idx = [(fold - 1 + i) % 10 for i in range(1)]
        traintestval.append([train_idx, val_idx, test_idx])

        if not os.path.exists(models_path + '/' + name0 + '/' + name1):
            os.makedirs(models_path + '/' + name0 + '/' + name1)

    joined_dataset, train_subsets, val_subsets, test_subsets = ApneaDataset.join_datasets(datasets, traintestval)
    joined_dataset.Zscore_normalization()

    data_analysis = joined_dataset.data_analysis()
    data_analysis += f"\nTrain: subsets {train_subsets}\nVal: subset {val_subsets}\nTest: subset {test_subsets}\n"
    joined_dataset.undersample_majority_class(0.0, train_subsets + val_subsets, prop = 1)
    data_analysis += "\nUNDERSAMPLING\n" + joined_dataset.data_analysis()

    joined_trainset = joined_dataset.get_subsets(train_subsets)
    joined_valset = joined_dataset.get_subsets(val_subsets)
    joined_testset = joined_dataset.get_subsets(test_subsets)
    data_analysis += f"\nTrain count: {len(joined_trainset)}\n\tWith apnea: {int(sum(sum((joined_trainset[:][1]).tolist(), [])))}\n\tWithout apnea: {len(joined_trainset) - int(sum(sum((joined_trainset[:][1]).tolist(), [])))}\nVal count: {len(joined_valset)}\n\tWith apnea: {int(sum(sum((joined_valset[:][1]).tolist(), [])))}\n\tWithout apnea: {len(joined_valset) - int(sum(sum((joined_valset[:][1]).tolist(), [])))}\nTest count: {len(joined_testset)}\n\tWith apnea: {int(sum(sum((joined_testset[:][1]).tolist(), [])))}\n\tWithout apnea: {len(joined_testset) - int(sum(sum((joined_testset[:][1]).tolist(), [])))}"

    input_size = joined_dataset.signal_len()
    model = Model(
            input_size = input_size,
            name = name1,
            n_filters_1 = 32,
            kernel_size_Conv1 = 3,
            n_filters_2 = 64,
            kernel_size_Conv2 = 3,
            n_filters_3 = 128,
            kernel_size_Conv3 = 3,
            n_filters_4 = 128,
            kernel_size_Conv4 = 3,
            n_filters_5 = 256,
            kernel_size_Conv5 = 3,
            n_filters_6 = 256,
            kernel_size_Conv6 = 3,
            dropout = 0,
            maxpool = 2,
        ).to(device)
    model.apply(init_weights)
    data_analysis += f"\nWeights initialized with nn.init.trunc_normal_(m.weight, std=0.02)\n"

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
        text = txt_file + data_analysis + model_arch,
        device = device)
    trainer.train(models_path + '/' + name0 + '/' + name1, verbose = True, plot = False, save_best_model = True)
    
    # test final with joined dataset
    tester = Tester(model = model,
                    testset = joined_testset,
                    batch_size = 32,
                    device = device, 
                    best_final = 'final')
    cm, metrics = tester.evaluate(models_path + '/' + name0 + '/' + name1, plot = False)

    # test best with joined dataset
    best_model = Model.load_model(models_path + '/' + name0 + '/' + name1, name1, input_size, best = True).to(device)
    best_tester = Tester(model = best_model,
                    testset = joined_testset,
                    batch_size = 32,
                    device = device, 
                    best_final = 'best')
    best_tester.evaluate(models_path + '/' + name0 + '/' + name1, plot = False)

    # test with each file
    for file in files: 
        ds = ApneaDataset.load_dataset(f"./data/ApneaDetection_HomePAPSignals/datasets/dataset2_archivo_1600{file:03d}.pth")
        if(ds._ApneaDataset__sr != 200):
            ds.resample_segments(200)
        
        train_idx = [(fold + i) % 10 for i in range(8)]
        val_idx = [(fold - 2 + i) % 10 for i in range(1)]
        test_idx = [(fold - 1 + i) % 10 for i in range(1)]
        
        testset = ds.get_subsets(test_idx)

        # test final with one dataset
        tester = Tester(model = model,
                    testset = testset,
                    batch_size = 32,
                    device = device, 
                    best_final = f'final_file_1600{file:03d}')
        tester.evaluate(models_path + '/' + name0 + '/' + name1, plot = False)
        # test best with joined dataset
        best_tester = Tester(model = best_model,
                    testset = testset,
                    batch_size = 32,
                    device = device, 
                    best_final = f'best_file_1600{file:03d}')
        best_tester.evaluate(models_path + '/' + name0 + '/' + name1, plot = False)
