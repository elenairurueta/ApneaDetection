from Imports import *
from itertools import product
from DataFormatting import ApneaDataset
from Training import Trainer
from Testing import Tester
from Models import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models_path = './models'

param_grid = {
    'overlap': [0, 10],
    'perc_apnea': [0.1, 0.3, 0.5, 0.7, 0.9]
}

param_combinations = list(product(param_grid['overlap'], param_grid['perc_apnea']))

results = []

for overlap, perc_apnea in param_combinations:
    
    torch.manual_seed(0)
    
    files = [4, 43, 53, 55, 63, 72, 84, 95, 105, 113, 122, 151] 

    path_annot = ""  #CHANGE
    path_edf = "" #CHANGE
    ApneaDataset.create_datasets(files, path_edf, path_annot, overlap, perc_apnea) 

    name0 = f'modelo_GS_overlap{overlap}_pa{int(perc_apnea)*100}'
    metrics_acum = {'Accuracy': [], 'Precision': [], 'Sensitivity': [], 'Specificity': [], 'F1': [], 'MCC': []}
    cm_acum = []
    best_metrics_acum = {'Accuracy': [], 'Precision': [], 'Sensitivity': [], 'Specificity': [], 'F1': [], 'MCC': []}
    best_cm_acum = []

    for fold in range(0,10):
        name1 = f'{name0}_fold{fold}'
        txt_archivo = ""
        
        datasets = []
        traintestval = []
        for file in files:
            txt_archivo += f"homepap-lab-full-1600{str(file).zfill(3)}\n"
            dataset = ApneaDataset.load_dataset(f".\data\ApneaDetection_HomePAPSignals\datasets\dataset_file_1600{file:03d}_overlap{overlap}_pa{int(perc_apnea*100)}.pth")

            if(dataset._ApneaDataset__sr != 200):
                dataset.resample_segments(200)
            datasets.append(dataset)
            
            train_idx = [(fold + i) % 10 for i in range(8)]
            val_idx = [(fold - 2 + i) % 10 for i in range(1)]
            test_idx = [(fold - 1 + i) % 10 for i in range(1)]
            traintestval.append([train_idx, val_idx, test_idx])

        if not os.path.exists(models_path + '/' + name0): 
            os.makedirs(models_path + '/' + name0) 

        joined_dataset, train_subsets, val_subsets, test_subsets = ApneaDataset.join_datasets(datasets, traintestval)
        joined_dataset.Zscore_normalization()

        analisis_datos = joined_dataset.data_analysis()
        analisis_datos += f"\nTrain: subsets {train_subsets}\nVal: subset {val_subsets}\nTest: subset {test_subsets}\n"
        joined_dataset.undersample_majority_class(0.0, train_subsets + val_subsets, prop = 1)
        analisis_datos += "\nUNDERSAMPLING\n" + joined_dataset.data_analysis()

        joined_trainset = joined_dataset.get_subsets(train_subsets)
        joined_valset = joined_dataset.get_subsets(val_subsets)
        joined_testset = joined_dataset.get_subsets(test_subsets)
        analisis_datos += f"\nTrain count: {len(joined_trainset)}\n\tWith apnea: {int(sum(sum((joined_trainset[:][1]).tolist(), [])))}\n\tWithout apnea: {len(joined_trainset) - int(sum(sum((joined_trainset[:][1]).tolist(), [])))}\nVal count: {len(joined_valset)}\n\tWith apnea: {int(sum(sum((joined_valset[:][1]).tolist(), [])))}\n\tWithout apnea: {len(joined_valset) - int(sum(sum((joined_valset[:][1]).tolist(), [])))}\nTest count: {len(joined_testset)}\n\tWith apnea: {int(sum(sum((joined_testset[:][1]).tolist(), [])))}\n\tWithout apnea: {len(joined_testset) - int(sum(sum((joined_testset[:][1]).tolist(), [])))}"
        
        input_size = joined_dataset.signal_len()
        model = Model(
            input_size = input_size,
            name = name1,
            n_filters_1 = 8,
            kernel_size_Conv1 = 35,
            n_filters_2 = 8,
            kernel_size_Conv2 = 50,
            n_filters_3 = 128,
            kernel_size_Conv3 = 35,
            n_filters_4 = 128,
            kernel_size_Conv4 = 35,
            n_filters_5 = 128,
            kernel_size_Conv5 = 50,
            n_filters_6 = 16,
            kernel_size_Conv6 = 50,
            dropout = 0.1,
            maxpool = 7
        ).to(device)

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
        model = trainer.train(models_path + '/' + name0, verbose = False, plot = False, save_model = False, save_best_model = False)
        
        tester = Tester(model = model,
                        testset = joined_testset,
                        batch_size = 32,
                        device = device, 
                        best_final = 'final')
        cm, metrics = tester.evaluate(models_path + '/' + name0, plot = False)
        cm_acum.append(cm)
        for key in metrics_acum:
            metrics_acum[key].append(float(metrics[key]))

    cm_acum = np.array(cm_acum)
    cm_mean = np.mean(cm_acum, axis=0)
    cm_std = np.std(cm_acum, axis=0)
    metrics_mean = {key: np.mean(metrics_acum[key]) for key in metrics_acum}
    metrics_std = {key: np.std(metrics_acum[key]) for key in metrics_acum}

    fig, ax = plt.subplots(figsize=(13, 6))
    cm_norm = cm_mean.astype('float') / cm_mean.sum(axis=1)[:, np.newaxis] * 100
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=['without apnea', 'with apnea'])
    cm_display.plot(cmap='Blues', ax=ax)
    for text in ax.texts:
        text.set_visible(False)
    for i in range(cm_mean.shape[0]):
        for j in range(cm_mean.shape[1]):
            text = ax.text(j, i, f'{cm_mean[i, j]:.2f} ± {cm_std[i, j]:.2f}', ha='center', va='center', color='black')
    ax.set_title("Confusion Matrix Final")

    metric_text = (f"Accuracy: ({metrics_mean['Accuracy']*100:.2f}±{metrics_std['Accuracy']*100:.2f})%\n"
                f"Precision: ({metrics_mean['Precision']*100:.2f}±{metrics_std['Precision']*100:.2f})%\n"
                f"Sensitivity: ({metrics_mean['Sensitivity']*100:.2f}±{metrics_std['Sensitivity']*100:.2f})%\n"
                f"Specificity: ({metrics_mean['Specificity']*100:.2f}±{metrics_std['Specificity']*100:.2f})%\n"
                f"F1: ({metrics_mean['F1']:.3f}±{metrics_std['F1']:.3f})\n"
                f"MCC: ({metrics_mean['MCC']:.3f}±{metrics_std['MCC']:.3f})")
    plt.gcf().text(0.1, 0.1, metric_text, ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    if os.path.exists(models_path):
        if not os.path.exists(models_path + '/' + name0): 
            os.makedirs(models_path + '/' + name0) 
        PATH = models_path + '/' + name0 + '/' + name0 + '_cm_metrics_mean_final.png'
        plt.savefig(PATH)
    plt.close()

    results.append({
        'overlap': overlap,
        'perc_apnea': perc_apnea,
        'Accuracy': f'{metrics_mean["Accuracy"]:.3f}±{metrics_std["Accuracy"]:.3f}',
        'Precision': f'{metrics_mean["Precision"]:.3f}±{metrics_std["Precision"]:.3f}',
        'Sensitivity': f'{metrics_mean["Sensitivity"]:.3f}±{metrics_std["Sensitivity"]:.3f}',
        'Specificity': f'{metrics_mean["Specificity"]:.3f}±{metrics_std["Specificity"]:.3f}',
        'F1': f'{metrics_mean["F1"]:.3f}±{metrics_std["F1"]:.3f}',
        'MCC': f'{metrics_mean["MCC"]:.3f}±{metrics_std["MCC"]:.3f}'
    })

df_results = pd.DataFrame(results)
df_results.to_excel('hyperparameter_search_results.xlsx', index=False)