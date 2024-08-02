from DataFormatting import ApneaDataset2
from Modelo import *
from Training import Trainer
from Testing import Tester
from Imports import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models_path = '/media/elena/Externo/models'

metrics_acum = {'Accuracy': [], 'Precision': [], 'Sensitivity': [], 'Specificity': [], 'F1': [], 'MCC': []}
cm_acum = []
best_metrics_acum = {'Accuracy': [], 'Precision': [], 'Sensitivity': [], 'Specificity': [], 'F1': [], 'MCC': []}
best_cm_acum = []

fold = 9
nombre0 = f'modelo_archivos_16000_4_43_63_72_84_95_1hp_fold{fold}'
txt_archivo = ""
archivos = [4, 43, 63, 72, 84, 95] 
datasets = []
traintestval = []

for archivo in archivos:
    txt_archivo += f"homepap-lab-full-1600{str(archivo).zfill(3)}\n"
    ds = ApneaDataset2.load_dataset(f"..\\data\\ApneaDetection_HomePAPSignals\\datasets\\dataset2_archivo_1600{archivo:03d}.pth")
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

for i in range(0,5): 

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
    trainer.train(models_path + f'/{nombre0}', verbose = True, plot = False, save_best_model = True)
    
    # test final
    tester = Tester(model = model,
                    testset = joined_testset,
                    batch_size = 32,
                    device = device, 
                    best_final = 'final')
    cm, metrics = tester.evaluate(models_path + f'/{nombre0}', plot = False)
    cm_acum.append(cm)
    for key in metrics_acum:
        metrics_acum[key].append(float(metrics[key]))

    # test best
    best_model = Model2.load_model(models_path + f'/{nombre0}', nombre, input_size, best = True).to(device)
    best_tester = Tester(model = best_model,
                    testset = joined_testset,
                    batch_size = 32,
                    device = device, 
                    best_final = 'best')
    best_cm, best_metrics = best_tester.evaluate(models_path + f'/{nombre0}', plot = False)
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
    if not os.path.exists(models_path + '/' + nombre0): 
        os.makedirs(models_path + '/' + nombre0) 
    PATH = models_path + '/' + nombre0 + '/' + nombre0 + '_cm_metrics_mean_final.png'
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
    if not os.path.exists(models_path + '/' + nombre0): 
        os.makedirs(models_path + '/' + nombre0) 
    PATH = models_path + '/' + nombre0 + '/' + nombre0 + '_cm_metrics_mean_best.png'
    plt.savefig(PATH)
plt.close()