from Imports import *
from DataFormatting import ApneaDataset
from Models import Model, init_weights
from Training import Trainer
from Testing import Tester
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models_path = '../models'

path_annot = "C:/Users/elena/OneDrive/Documentos/TFG/Dataset/HomePAP/polysomnography/annotations-events-profusion/lab/full"  #CHANGE
path_edf = "C:/Users/elena/OneDrive/Documentos/TFG/Dataset/HomePAP/polysomnography/edfs/lab/full" #CHANGE

# Get all files in the specified directories
files = [int(''.join(filter(str.isdigit, f))) for f in os.listdir(path_edf) if f.endswith('.edf')]

ApneaDataset.create_datasets(files, path_edf, path_annot, overlap = 10, perc_apnea = 0.3, filtering = True, filter = "FIR_Notch")


metrics_acum_total = {'Accuracy': [], 'Precision': [], 'Sensitivity': [], 'Specificity': [], 'F1': [], 'MCC': []}
cm_acum_total = []
metrics_std_total = {'Accuracy': [], 'Precision': [], 'Sensitivity': [], 'Specificity': [], 'F1': [], 'MCC': []}
cm_std_total = []
best_metrics_acum_total = {'Accuracy': [], 'Precision': [], 'Sensitivity': [], 'Specificity': [], 'F1': [], 'MCC': []}
best_cm_acum_total = []
best_metrics_std_total = {'Accuracy': [], 'Precision': [], 'Sensitivity': [], 'Specificity': [], 'F1': [], 'MCC': []}
best_cm_std_total = []

txt_file = ""
name0 = f'model_allfiles'
if not os.path.exists(models_path + '/' + name0):
    os.makedirs(models_path + '/' + name0)
        
for fold in range(0,9):        

    metrics_acum = {'Accuracy': [], 'Precision': [], 'Sensitivity': [], 'Specificity': [], 'F1': [], 'MCC': []}
    cm_acum = []
    best_metrics_acum = {'Accuracy': [], 'Precision': [], 'Sensitivity': [], 'Specificity': [], 'F1': [], 'MCC': []}
    best_cm_acum = []

    name1 = name0 + f'_fold{fold}'
    datasets = []
    traintestval = []

    for file in files: 
        txt_file += f"homepap-lab-full-{str(file)}\n"
        try:
            ds = ApneaDataset.load_dataset(f"../data/ApneaDetection_HomePAPSignals/datasets/dataset_file_{file}.pth")
        except:
            continue
        if(ds._ApneaDataset__sr != 200):
            ds.resample_segments(200)
        datasets.append(ds)
        
        train_idx = [(fold + i) % 9 for i in range(8)]  
        val_idx = [(fold + 8) % 9]
        test_idx = [9]  
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

    if not os.path.exists(models_path + '/' + name0 + '/' + name1):
        os.makedirs(models_path + '/' + name0 + '/' + name1)

    for i in range(0,5):
        name2 = name1 + f'_{i}'
        
        input_size = joined_dataset.signal_len()
        model = Model(
                input_size = input_size,
                name = name2,
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
            text = txt_file + data_analysis + model_arch,
            device = device)
        trainer.train(models_path + '/' + name0 + '/' + name1, verbose = True, plot = False, save_best_model = True) # + '/' + name2
        
        # test final with joined dataset
        tester = Tester(model = model,
                        testset = joined_testset,
                        batch_size = 32,
                        device = device, 
                        best_final = 'final')
        cm, metrics = tester.evaluate(models_path + '/' + name0 + '/' + name1, plot = False) # + '/' + name2
        cm_acum.append(cm)
        for key in metrics_acum:
            metrics_acum[key].append(float(metrics[key]))

        # test best with joined dataset
        best_model = Model.load_model(models_path + '/' + name0 + '/' + name1, name2, # + '/' + name2
                input_size, best = True,                 
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
        best_tester = Tester(model = best_model,
                        testset = joined_testset,
                        batch_size = 32,
                        device = device, 
                        best_final = 'best')
        best_cm, best_metrics = best_tester.evaluate(models_path + '/' + name0 + '/' + name1, plot = False) # + '/' + name2
        best_cm_acum.append(best_cm)
        for key in best_metrics_acum:
            best_metrics_acum[key].append(float(best_metrics[key]))

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
        if not os.path.exists(models_path + '/' + name0 + '/' + name1): 
            os.makedirs(models_path + '/' + name0 + '/' + name1) 
        PATH = models_path + '/' + name0 + '/' + name1 + '/' + name1 + '_cm_metrics_mean_final.png'
        plt.savefig(PATH)
    plt.close()


    best_cm_acum = np.array(best_cm_acum)
    best_cm_mean = np.mean(best_cm_acum, axis=0)
    best_cm_std = np.std(best_cm_acum, axis=0)
    best_metrics_mean = {key: np.mean(best_metrics_acum[key]) for key in best_metrics_acum}
    best_metrics_std = {key: np.std(best_metrics_acum[key]) for key in best_metrics_acum}

    fig, ax = plt.subplots(figsize=(13, 6))
    best_cm_norm = best_cm_mean.astype('float') / best_cm_mean.sum(axis=1)[:, np.newaxis] * 100
    best_cm_display = ConfusionMatrixDisplay(confusion_matrix=best_cm_norm, display_labels=['without apnea', 'with apnea'])
    best_cm_display.plot(cmap='Blues', ax=ax)
    for text in ax.texts:
        text.set_visible(False)
    for i in range(best_cm_mean.shape[0]):
        for j in range(best_cm_mean.shape[1]):
            text = ax.text(j, i, f'{best_cm_mean[i, j]:.2f} ± {best_cm_std[i, j]:.2f}', ha='center', va='center', color='black')
    ax.set_title("Confusion Matrix Best")

    best_metric_text = (f"Accuracy: ({best_metrics_mean['Accuracy']*100:.2f}±{best_metrics_std['Accuracy']*100:.2f})%\n"
                f"Precision: ({best_metrics_mean['Precision']*100:.2f}±{best_metrics_std['Precision']*100:.2f})%\n"
                f"Sensitivity: ({best_metrics_mean['Sensitivity']*100:.2f}±{best_metrics_std['Sensitivity']*100:.2f})%\n"
                f"Specificity: ({best_metrics_mean['Specificity']*100:.2f}±{best_metrics_std['Specificity']*100:.2f})%\n"
                f"F1: ({best_metrics_mean['F1']:.3f}±{best_metrics_std['F1']:.3f})\n"
                f"MCC: ({best_metrics_mean['MCC']:.3f}±{best_metrics_std['MCC']:.3f})")
    plt.gcf().text(0.1, 0.1, best_metric_text, ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))


    if os.path.exists(models_path):
        if not os.path.exists(models_path + '/' + name0 + '/' + name1): 
            os.makedirs(models_path + '/' + name0 + '/' + name1) 
        PATH = models_path + '/' + name0 + '/' + name1 + '/' + name1 + '_cm_metrics_mean_best.png'
        plt.savefig(PATH)
    plt.close()

    cm_acum_total.append(cm_mean)
    cm_std_total.append(cm_std)
    for key in metrics_acum_total:
        metrics_acum_total[key].append(float(metrics_mean[key]))
        metrics_std_total[key].append(float(metrics_std[key]))

    best_cm_acum_total.append(best_cm_mean)
    best_cm_std_total.append(best_cm_std)
    for key in best_metrics_acum_total:
        best_metrics_acum_total[key].append(float(best_metrics_mean[key]))
        best_metrics_std_total[key].append(float(best_metrics_std[key]))


mean_cm_acum_np = np.array(cm_acum_total)
std_cm_acum_np = np.array(cm_std_total)

mean_final = np.mean(mean_cm_acum_np, axis=0)
std_final = np.sqrt(np.mean(std_cm_acum_np**2, axis=0) + np.var(mean_cm_acum_np, axis=0))

fig, ax = plt.subplots(figsize=(13, 6))
cm_norm = mean_final.astype('float') / mean_final.sum(axis=1)[:, np.newaxis] * 100
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=['without apnea', 'with apnea'])
cm_display.plot(cmap='Blues', ax=ax)
for text in ax.texts:
    text.set_visible(False)
for i in range(mean_final.shape[0]):
    for j in range(mean_final.shape[1]):
        text = ax.text(j, i, f'{mean_final[i, j]:.2f} ± {std_final[i, j]:.2f}', ha='center', va='center', color='black')
ax.set_title("Confusion Matrix Final")

mean_metrics = {}
std_metrics = {}
for metric in metrics_acum_total.keys():
    mean_values = np.array(metrics_acum_total[metric])
    std_values = np.array(metrics_std_total[metric])
    mean_final = np.mean(mean_values)
    std_final = np.sqrt(np.mean(std_values**2) + np.var(mean_values))
    mean_metrics[metric] = mean_final
    std_metrics[metric] = std_final

metric_text = (f"Accuracy: ({mean_metrics['Accuracy']*100:.2f}±{std_metrics['Accuracy']*100:.2f})%\n"
            f"Precision: ({mean_metrics['Precision']*100:.2f}±{std_metrics['Precision']*100:.2f})%\n"
            f"Sensitivity: ({mean_metrics['Sensitivity']*100:.2f}±{std_metrics['Sensitivity']*100:.2f})%\n"
            f"Specificity: ({mean_metrics['Specificity']*100:.2f}±{std_metrics['Specificity']*100:.2f})%\n"
            f"F1: ({mean_metrics['F1']:.3f}±{std_metrics['F1']:.3f})\n"
            f"MCC: ({mean_metrics['MCC']:.3f}±{std_metrics['MCC']:.3f})")
plt.gcf().text(0.1, 0.1, metric_text, ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

if os.path.exists(models_path):
    if not os.path.exists(models_path + '/' + name0 + '/FINAL_CrossVal7'): 
        os.makedirs(models_path + '/' + name0 + '/FINAL_CrossVal7') 
    PATH = models_path + '/' + name0 + '/FINAL_CrossVal7/' + 'FINAL_CrossVal7_cm_metrics_mean_final.png'
    plt.savefig(PATH)
plt.close()


mean_cm_acum_np = np.array(best_cm_acum_total)
std_cm_acum_np = np.array(best_cm_std_total)

mean_final = np.mean(mean_cm_acum_np, axis=0)
std_final = np.sqrt(np.mean(std_cm_acum_np**2, axis=0) + np.var(mean_cm_acum_np, axis=0))

fig, ax = plt.subplots(figsize=(13, 6))
cm_norm = mean_final.astype('float') / mean_final.sum(axis=1)[:, np.newaxis] * 100
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=['without apnea', 'with apnea'])
cm_display.plot(cmap='Blues', ax=ax)
for text in ax.texts:
    text.set_visible(False)
for i in range(mean_final.shape[0]):
    for j in range(mean_final.shape[1]):
        text = ax.text(j, i, f'{mean_final[i, j]:.2f} ± {std_final[i, j]:.2f}', ha='center', va='center', color='black')
ax.set_title("Confusion Matrix Best")

mean_metrics = {}
std_metrics = {}
for metric in best_metrics_acum_total.keys():
    mean_values = np.array(best_metrics_acum_total[metric])
    std_values = np.array(best_metrics_std_total[metric])
    mean_final = np.mean(mean_values)
    std_final = np.sqrt(np.mean(std_values**2) + np.var(mean_values))
    mean_metrics[metric] = mean_final
    std_metrics[metric] = std_final

metric_text = (f"Accuracy: ({mean_metrics['Accuracy']*100:.2f}±{std_metrics['Accuracy']*100:.2f})%\n"
            f"Precision: ({mean_metrics['Precision']*100:.2f}±{std_metrics['Precision']*100:.2f})%\n"
            f"Sensitivity: ({mean_metrics['Sensitivity']*100:.2f}±{std_metrics['Sensitivity']*100:.2f})%\n"
            f"Specificity: ({mean_metrics['Specificity']*100:.2f}±{std_metrics['Specificity']*100:.2f})%\n"
            f"F1: ({mean_metrics['F1']:.3f}±{std_metrics['F1']:.3f})\n"
            f"MCC: ({mean_metrics['MCC']:.3f}±{std_metrics['MCC']:.3f})")
plt.gcf().text(0.1, 0.1, metric_text, ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))


if os.path.exists(models_path):
    if not os.path.exists(models_path + '/' + name0 + '/FINAL_CrossVal7'): 
        os.makedirs(models_path + '/' + name0 + '/FINAL_CrossVal7') 
    PATH = models_path + '/' + name0 + '/FINAL_CrossVal7/' + 'FINAL_CrossVal7_cm_metrics_mean_best.png'
    plt.savefig(PATH)
plt.close()