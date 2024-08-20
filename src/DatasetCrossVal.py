from DataFormatting import ApneaDataset2
from Training import Trainer
from Testing import Tester
from Imports import *


class Model(nn.Module):
    """Neural network model class"""
    def __init__(self, input_size, nombre:str, n_filters_1=32, kernel_size_Conv1=3, n_filters_2=64, kernel_size_Conv2=3, n_filters_3=128, kernel_size_Conv3=3, n_filters_4=128, kernel_size_Conv4=3, n_filters_5=256, kernel_size_Conv5=3, n_filters_6=256, kernel_size_Conv6=3, dropout=0.5, maxpool=2):
        """
        Initializes the neural network model.

        Args:
            input_size (int): size of the input data.
            nombre (str): name of the model.
        """

        super().__init__()
        self.nombre = nombre

        # Calculate output size after convolutions and pooling
        def conv_output_size(input_size, kernel_size, padding, stride, dilation = 1):
            return int(np.floor((input_size + 2*padding - dilation*(kernel_size - 1) - 1)/stride + 1))
        
        def maxpool_output_size(input_size, kernel_size, stride, padding = 0, dilation = 1):
            return int(np.floor((input_size + 2*padding - dilation*(kernel_size - 1) - 1)/stride + 1))

        # Compute the size after each layer
        size = input_size
        # size = conv_output_size(size, kernel_size_Conv1, padding=1, stride=1)
        # size = conv_output_size(size, kernel_size_Conv2, padding=1, stride=1)
        size = maxpool_output_size(size, kernel_size=maxpool, stride=maxpool)
        # size = conv_output_size(size, kernel_size_Conv3, padding=1, stride=1)
        # size = conv_output_size(size, kernel_size_Conv4, padding=1, stride=1)
        size = maxpool_output_size(size, kernel_size=maxpool, stride=maxpool)
        # size = conv_output_size(size, kernel_size_Conv5, padding=1, stride=1)
        # size = conv_output_size(size, kernel_size_Conv6, padding=1, stride=1)
        size = maxpool_output_size(size, kernel_size=maxpool, stride=maxpool)

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, n_filters_1, kernel_size=kernel_size_Conv1, stride=1, padding='same'),
            nn.ReLU(), 
            nn.Conv1d(n_filters_1, n_filters_2, kernel_size=kernel_size_Conv2, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(n_filters_2),
            nn.MaxPool1d(kernel_size=maxpool, stride=maxpool),
            nn.Dropout(dropout),

            nn.Conv1d(n_filters_2, n_filters_3, kernel_size=kernel_size_Conv3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv1d(n_filters_3, n_filters_4, kernel_size=kernel_size_Conv4, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(n_filters_4),
            nn.MaxPool1d(kernel_size=maxpool, stride=maxpool),
            nn.Dropout(dropout),

            nn.Conv1d(n_filters_4, n_filters_5, kernel_size=kernel_size_Conv5, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv1d(n_filters_5, n_filters_6, kernel_size=kernel_size_Conv6, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(n_filters_6),
            nn.MaxPool1d(kernel_size=maxpool, stride=maxpool),
            nn.Dropout(dropout)
        )

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(n_filters_6 * size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Defines the forward pass of the neural network model.
        """
        x = self.conv_layers(x)
        batch_size = x.size(0)
        x = x.view(batch_size, -1) 
        x = self.fc_layers(x)
        return x
    
    def get_nombre(self):
        """
        Args: none.

        Returns:
            - str: the name of the model.
        """
        return self.nombre

    def get_architecture(self):
        """
        Args: none.

        Returns:
            - str: the architecture of the model.
        """
        return '\n\n' + str(self)

    def save_model(self, models_path, extension:str = '.pth'):
        """
        Saves the parameters of the model to a file.

        Args:
            - extension (str, optional): extension of the file ('.pt' or '.pth'). Defaults to '.pth'.

        Returns: none
        """
        if os.path.exists(models_path):
            if not os.path.exists(models_path + f'/{self.nombre}'):
                os.makedirs(models_path + f'/{self.nombre}')
            PATH = models_path + f'/{self.nombre}/{self.nombre + extension}'             
        torch.save(self.state_dict(), PATH)

    @staticmethod
    def load_model(models_path, nombre, input_size, n_filters_1=32, kernel_size_Conv1=3, n_filters_2=64, kernel_size_Conv2=3, n_filters_3=128, kernel_size_Conv3=3, n_filters_4=128, kernel_size_Conv4=3, n_filters_5=256, kernel_size_Conv5=3, n_filters_6=256, kernel_size_Conv6=3, dropout=0.5, maxpool=2, extension:str = '.pth', best = False):
        """
        Loads a pre-trained model from a file.

        Args:
            - nombre (str): name of the model.
            - input_size (int): size of the input data.
            - extension (str, optional): extension of the file ('.pt' or '.pth'). Defaults to '.pth'.
            
        Returns:
            - Model: the loaded model.
        """
        model = Model(input_size, nombre, n_filters_1, kernel_size_Conv1, n_filters_2, kernel_size_Conv2, n_filters_3, kernel_size_Conv3, n_filters_4, kernel_size_Conv4, n_filters_5, kernel_size_Conv5, n_filters_6, kernel_size_Conv6, dropout, maxpool)
        if(best):
            PATH = models_path + f'/{nombre}/{nombre}_best{extension}' 
            model.load_state_dict(torch.load(PATH))
        else:
            PATH = models_path + f'/{nombre}/{nombre + extension}'
            model.load_state_dict(torch.load(PATH))
        model.eval()
        return model



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models_path = '/media/elena/Externo/models'

metrics_acum_total = {'Accuracy': [], 'Precision': [], 'Sensitivity': [], 'Specificity': [], 'F1': [], 'MCC': []}
cm_acum_total = []
metrics_std_total = {'Accuracy': [], 'Precision': [], 'Sensitivity': [], 'Specificity': [], 'F1': [], 'MCC': []}
cm_std_total = []
best_metrics_acum_total = {'Accuracy': [], 'Precision': [], 'Sensitivity': [], 'Specificity': [], 'F1': [], 'MCC': []}
best_cm_acum_total = []
best_metrics_std_total = {'Accuracy': [], 'Precision': [], 'Sensitivity': [], 'Specificity': [], 'F1': [], 'MCC': []}
best_cm_std_total = []

for fold in range(0,10):

    metrics_acum = {'Accuracy': [], 'Precision': [], 'Sensitivity': [], 'Specificity': [], 'F1': [], 'MCC': []}
    cm_acum = []
    best_metrics_acum = {'Accuracy': [], 'Precision': [], 'Sensitivity': [], 'Specificity': [], 'F1': [], 'MCC': []}
    best_cm_acum = []

    nombre0 = f'modelo_archivos_16000_4_43_53_55_63_72_84_95_105_113_122_151_fold{fold}'
    txt_archivo = ""
    archivos = [4, 43, 53, 55, 63, 72, 84, 95, 105, 113, 122, 151] 
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
    joined_dataset.Zscore_normalization()

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
        model = Model(
            input_size = input_size,
            nombre = nombre,
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
        best_model = Model.load_model(models_path + f'/{nombre0}', nombre, input_size, 
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
                                      maxpool = 7,
                                      best = True).to(device)
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

models_path = '/media/elena/Externo/models'

if os.path.exists(models_path):
    if not os.path.exists(models_path + '/FINAL_CrossVal4'): 
        os.makedirs(models_path + '/FINAL_CrossVal4') 
    PATH = models_path + '/FINAL_CrossVal4/' + 'FINAL_CrossVal4_cm_metrics_mean_final.png'
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

models_path = '/media/elena/Externo/models'

if os.path.exists(models_path):
    if not os.path.exists(models_path + '/FINAL_CrossVal4'): 
        os.makedirs(models_path + '/FINAL_CrossVal4') 
    PATH = models_path + '/FINAL_CrossVal4/' + 'FINAL_CrossVal4_cm_metrics_mean_best.png'
    plt.savefig(PATH)
plt.close()
