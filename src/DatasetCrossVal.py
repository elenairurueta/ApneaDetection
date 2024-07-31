from DataFormatting import ApneaDataset2
from Modelo import *
from Training import Trainer
from Testing import Tester
from Imports import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cm_acum = []
metrics_acum = []
cm00 = []
cm10 = []
cm11 = []
cm01 = []
acc = []
prec = []
sens = []
spec = []
F1 = []
mcc = []

nombre0 = 'modelo_archivos_16000_4_43_63_72_84_95_1hp_fold5'
txt_archivo = ""
archivos = [4, 43, 63, 72, 84, 95] #
datasets = []
traintestval = []
for archivo in archivos:
    txt_archivo += f"homepap-lab-full-1600{str(archivo).zfill(3)}\n"
    ds = ApneaDataset2.load_dataset(f"..\\data\\ApneaDetection_HomePAPSignals\\datasets\\dataset2_archivo_1600{archivo:03d}.pth")
    if(ds._ApneaDataset2__sr != 200):
        ds.resample_segments(200)
    datasets.append(ds)
    traintestval += [[[5, 6, 7, 8, 9, 0, 1, 2], [3], [4]]] ## train, val, test

for i in range(0,5):
    joined_dataset, train_subsets, val_subsets, test_subsets = ApneaDataset2.join_datasets(datasets, traintestval)
    analisis_datos = joined_dataset.analisis_datos()
    analisis_datos += f"\nTrain: subsets {train_subsets}\nVal: subset {val_subsets}\nTest: subset {test_subsets}\n"
    joined_dataset.undersample_majority_class(0.0, train_subsets + val_subsets, prop = 1)
    analisis_datos += "\nUNDERSAMPLING\n" + joined_dataset.analisis_datos()

    joined_trainset = joined_dataset.get_subsets(train_subsets)
    joined_valset = joined_dataset.get_subsets(val_subsets)
    joined_testset = joined_dataset.get_subsets(test_subsets)
    analisis_datos += f"\nTrain count: {len(joined_trainset)}\n\tWith apnea: {int(sum(sum((joined_trainset[:][1]).tolist(), [])))}\n\tWithout apnea: {len(joined_trainset) - int(sum(sum((joined_trainset[:][1]).tolist(), [])))}\nVal count: {len(joined_valset)}\n\tWith apnea: {int(sum(sum((joined_valset[:][1]).tolist(), [])))}\n\tWithout apnea: {len(joined_valset) - int(sum(sum((joined_valset[:][1]).tolist(), [])))}\nTest count: {len(joined_testset)}\n\tWith apnea: {int(sum(sum((joined_testset[:][1]).tolist(), [])))}\n\tWithout apnea: {len(joined_testset) - int(sum(sum((joined_testset[:][1]).tolist(), [])))}"

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
    trainer.train(verbose = True, plot = False, save_best_model = True)
    
    best_model = Model2.load_model(nombre, input_size, best = True).to(device)
    tester = Tester(model = best_model,
                    testset = joined_testset,
                    batch_size = 32,
                    device = device)
    cm, metrics = tester.evaluate(plot = False)

    cm_acum.append(cm)
    metrics_acum.append(metrics)
    cm00.append(cm[0][0])
    cm01.append(cm[0][1])
    cm10.append(cm[1][0])
    cm11.append(cm[1][1])
    acc.append(float(metrics['Accuracy']))
    prec.append(float(metrics['Precision']))
    sens.append(float(metrics['Sensitivity']))
    spec.append(float(metrics['Specificity']))
    F1.append(float(metrics['F1']))
    mcc.append(float(metrics['MCC']))

mean00 = np.mean(cm00)
std00 = np.std(cm00)
mean01 = np.mean(cm01)
std01 = np.std(cm01)
mean10 = np.mean(cm10)
std10 = np.std(cm10)
mean11 = np.mean(cm11)
std11 = np.std(cm11)
meanacc = np.mean(acc)
stdacc = np.std(acc)
meanprec = np.mean(prec)
stdprec = np.std(prec)
meansens = np.mean(sens)
stdsens = np.std(sens)
meanspec = np.mean(spec)
stdspec = np.std(spec)
meanF1 = np.mean(F1)
stdF1 = np.std(F1)
meanMCC = np.mean(mcc)
stdMCC = np.std(mcc)

cm_mean = np.array([[mean00, mean01], [mean10, mean11]])
cm_std = np.array([[std00, std01], [std10, std11]])
fig, ax = plt.subplots(figsize=(13, 6))
cm_norm = cm_mean.astype('float') / cm_mean.sum(axis=1)[:, np.newaxis] * 100
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=['without apnea', 'with apnea'])
cm_display.plot(cmap='Blues', ax=ax)
for text in ax.texts:
    text.set_visible(False)
for i in range(cm_mean.shape[0]):
    for j in range(cm_mean.shape[1]):
        text = ax.text(j, i, f'{cm_mean[i, j]:.2f} +- {cm_std[i, j]:.2f}', ha='center', va='center', color='black')
ax.set_title("Confusion Matrix")

metric_text = (f"Accuracy: ({meanacc*100:.2f}+-{stdacc*100:.2f})%" +
            f"\nPrecision: ({meanprec*100:.2f}+-{stdprec*100:.2f})%" +
            f"\nSensitivity: ({meansens*100:.2f}+-{stdsens*100:.2f})%" +
            f"\nSpecificity: ({meanspec*100:.2f}+-{stdspec*100:.2f})%" +
            f"\nF1: ({meanF1:.3f}+-{stdF1:.3f})"
            f"\nMCC: ({meanMCC:.3f}+-{stdMCC:.3f})" )
plt.gcf().text(0.1, 0.1, metric_text, ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

if os.path.exists(f'/home/elena/Desktop/models'):
    if not os.path.exists(f'/home/elena/Desktop/models/' + nombre0): 
        os.makedirs(f'/home/elena/Desktop/models/' + nombre0) 
    PATH = f'/home/elena/Desktop/models/' + nombre0 + '/' + nombre0 + '_cm_metrics_mean.png'
    plt.savefig(PATH)

plt.close()





### 
# if os.path.exists(f'D:/models'):
#     if not os.path.exists(f'D:/models/modelo_archivos_16000_4_43_63_72_84_95_1hp_fold5'): ##CAMBIAR
#         os.makedirs(f'D:/models/modelo_archivos_16000_4_43_63_72_84_95_1hp_fold5') ##CAMBIAR
#     PATH = f'D:/models/modelo_archivos_16000_4_43_63_72_84_95_1hp_fold5/modelo_archivos_16000_4_43_63_72_84_95_1hp_fold5_cm_metrics_mean.png' ##CAMBIAR
# elif os.path.exists(f'./models'):
#     if not os.path.exists(f'./models/modelo_archivos_16000_4_43_63_72_84_95_1hp_fold5'): ##CAMBIAR
#         os.makedirs(f'../models/modelo_archivos_16000_4_43_63_72_84_95_1hp_fold5') ##CAMBIAR
#     PATH = f'./models/modelo_archivos_16000_4_43_63_72_84_95_1hp_fold5/modelo_archivos_16000_4_43_63_72_84_95_1hp_fold5_cm_metrics_mean.png' ##CAMBIAR
# plt.savefig(PATH)
# if os.path.exists(f'/home/elena/media/disk/_cygdrive_D_models'):
#     if not os.path.exists(f'/home/elena/media/disk/_cygdrive_D_models/' + nombre0): 
#         os.makedirs(f'/home/elena/media/disk/_cygdrive_D_models/' + nombre0) 
#     PATH = f'/home/elena/media/disk/_cygdrive_D_models/' + nombre0 + '/' + nombre0 + '_cm_metrics_mean.png'
#     plt.savefig(PATH)