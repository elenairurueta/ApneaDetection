from DataFormatting import ApneaDataset
from Training import Trainer
from Models import Model
from Imports import *

def plot_weights(title, norm = False, save = True):
    """
    This function visualizes the weights of the convolutional and fully connected layers.
    """
    fig, axes = plt.subplots(3, 3, figsize=(12, 6))
    fig.suptitle(title)
    idx = 0
    for i, layer in enumerate(model.conv_layers + model.fc_layers):
        if isinstance(layer, nn.Conv1d):
            k = layer.groups / (layer.in_channels * layer.kernel_size[0])
            title1 = f'Layer {idx + 1}: Conv1D'
        elif isinstance(layer, nn.Linear):
            k = 1/layer.in_features
            title1 = f'Layer {idx + 1}: Linear'
        else:
            continue
            
        weights = layer.weight.data.cpu().flatten()
        
        ax = axes.flat[idx]

        ax.hist(weights.numpy(), bins=100, color='b', alpha=0.7)
        if(norm):
            w = torch.empty(weights.shape)
            nn.init.uniform_(w, -np.sqrt(k), np.sqrt(k))
            ax.hist(w.numpy(), bins=100, color='r', alpha=0.5, histtype='step')

        ax.set_title(title1) 
        ax.set_xlabel('Pesos')
        ax.set_ylabel('Frecuencia')
        ax.grid(True)
        idx += 1

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if os.path.exists(models_path):
        if not os.path.exists(models_path + f'/{nombre}'): 
            os.makedirs(models_path + f'/{nombre}') 
        PATH = models_path + f'/{nombre}/{nombre}_weights_{title}.png'
        plt.savefig(PATH)
    #plt.show()
    plt.close()
    
def init_weights(m):
    """
    Initializes the weights of the model's layers using a truncated normal distribution for the weights and zero for biases.
    """
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models_path = './models'

fold = 0
nombre0 = f'modelo_prueba'
txt_archivo = ""
archivos = [4, 43, 53, 55, 63, 72, 84, 95, 105, 113, 122, 151] 
datasets = []
traintestval = []

for archivo in archivos:
    txt_archivo += f"homepap-lab-full-1600{str(archivo).zfill(3)}\n"
    ds = ApneaDataset.load_dataset(f".\data\ApneaDetection_HomePAPSignals\datasets\dataset2_archivo_1600{archivo:03d}.pth")
    if(ds._ApneaDataset__sr != 200):
        ds.resample_segments(200)
    datasets.append(ds)
    
    train_idx = [(fold + i) % 10 for i in range(8)]
    val_idx = [(fold - 2 + i) % 10 for i in range(1)]
    test_idx = [(fold - 1 + i) % 10 for i in range(1)]
    traintestval.append([train_idx, val_idx, test_idx])

if not os.path.exists(models_path + '/' + nombre0): 
    os.makedirs(models_path + '/' + nombre0) 

joined_dataset, train_subsets, val_subsets, test_subsets = ApneaDataset.join_datasets(datasets, traintestval)

analisis_datos = joined_dataset.data_analysis()
analisis_datos += f"\nTrain: subsets {train_subsets}\nVal: subset {val_subsets}\nTest: subset {test_subsets}\n"
joined_dataset.undersample_majority_class(0.0, train_subsets + val_subsets, prop = 1)
analisis_datos += "\nUNDERSAMPLING\n" + joined_dataset.data_analysis()
joined_trainset = joined_dataset.get_subsets(train_subsets)
joined_valset = joined_dataset.get_subsets(val_subsets)
joined_testset = joined_dataset.get_subsets(test_subsets)
analisis_datos += f"\nTrain count: {len(joined_trainset)}\n\tWith apnea: {int(sum(sum((joined_trainset[:][1]).tolist(), [])))}\n\tWithout apnea: {len(joined_trainset) - int(sum(sum((joined_trainset[:][1]).tolist(), [])))}\nVal count: {len(joined_valset)}\n\tWith apnea: {int(sum(sum((joined_valset[:][1]).tolist(), [])))}\n\tWithout apnea: {len(joined_valset) - int(sum(sum((joined_valset[:][1]).tolist(), [])))}\nTest count: {len(joined_testset)}\n\tWith apnea: {int(sum(sum((joined_testset[:][1]).tolist(), [])))}\n\tWithout apnea: {len(joined_testset) - int(sum(sum((joined_testset[:][1]).tolist(), [])))}"

nombre = nombre0

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
plot_weights('BeforeInitializing', save = True)
model.apply(init_weights)
plot_weights('AfterInitializingBeforeTraining', save = True)

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
plot_weights('AfterTraining', save = True)