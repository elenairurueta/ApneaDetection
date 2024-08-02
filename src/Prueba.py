from Modelo import Model2
from DataFormatting import ApneaDataset2
from Testing import Tester
from Imports import *

models_path = '/media/elena/Externo/models'

for fold in [8, 9]:
    nombre0 = f'modelo_archivos_16000_4_43_63_72_84_95_1hp_fold{fold}'
    metrics_acum = {'Accuracy': [], 'Precision': [], 'Sensitivity': [], 'Specificity': [], 'F1': [], 'MCC': []}
    cm_acum = []

    archivos = [4, 43, 63, 72, 84, 95] 
    datasets = []
    traintestval = []

    for archivo in archivos:
        ds = ApneaDataset2.load_dataset(f"..\\data\\ApneaDetection_HomePAPSignals\\datasets\\dataset2_archivo_1600{archivo:03d}.pth")
        if(ds._ApneaDataset2__sr != 200):
            ds.resample_segments(200)
        datasets.append(ds)
        
        train_idx = [(fold + i) % 10 for i in range(8)]
        val_idx = [(fold - 1 + i) % 10 for i in range(1)]
        test_idx = [(fold - 2 + i) % 10 for i in range(1)]
        traintestval.append([train_idx, val_idx, test_idx])
    
    joined_dataset, train_subsets, val_subsets, test_subsets = ApneaDataset2.join_datasets(datasets, traintestval)
    joined_dataset.undersample_majority_class(0.0, train_subsets + val_subsets, prop = 1)
    joined_trainset = joined_dataset.get_subsets(train_subsets)
    joined_valset = joined_dataset.get_subsets(val_subsets)
    joined_testset = joined_dataset.get_subsets(test_subsets)
    input_size = joined_dataset.signal_len()

    for i in range(0,5):
        nombre = nombre0 + f'_{i}'
        model = Model2.load_model(models_path + f'/{nombre0}', nombre, input_size)
        tester = Tester(model = model,
                    testset = joined_testset,
                    batch_size = 32,
                    device = "cpu")
        cm, metrics = tester.evaluate(models_path + f'/{nombre0}', plot = False)
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
    ax.set_title("Confusion Matrix")

    metric_text = (f"Accuracy: ({metrics_mean['Accuracy']*100:.2f}±{metrics_std['Accuracy']*100:.2f})%\n"
                f"Precision: ({metrics_mean['Precision']*100:.2f}±{metrics_std['Precision']*100:.2f})%\n"
                f"Sensitivity: ({metrics_mean['Sensitivity']*100:.2f}±{metrics_std['Sensitivity']*100:.2f})%\n"
                f"Specificity: ({metrics_mean['Specificity']*100:.2f}±{metrics_std['Specificity']*100:.2f})%\n"
                f"F1: ({metrics_mean['F1']:.3f}±{metrics_std['F1']:.3f})\n"
                f"MCC: ({metrics_mean['MCC']:.3f}±{metrics_std['MCC']:.3f})")
    plt.gcf().text(0.1, 0.1, metric_text, ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    if os.path.exists(models_path):
        if not os.path.exists(models_path + '/' + nombre0): 
            os.makedirs(models_path + '/' + nombre0) 
        PATH = models_path + '/' + nombre0 + '/' + nombre0 + '_cm_metrics_mean.png'
        plt.savefig(PATH)

    plt.close()




