# ApneaDetection
## Overview
This project is the final assignment for my Biomedical Engineering degree. It involves developing a machine learning model based on Convolutional Neural Networks (CNN) to detect sleep apnea in EEG signals.
## Initial Version
In the initial version of this project, simulated signals with characteristics similar to real EEG signals were used to train and test the model.
## Second Version
In the second version of this project, HomePAP Database EEG signals from polysomnography studies were used to train and test the model. More information about these signals can be found in the Docs folder.

## Installation

Clone the repository:
```bash
git clone https://github.com/elenairurueta/ApneaDetection.git
```
Create environment and install the required packages:
```bash
conda create --name ApneaDetection --file requirements.txt
```

If pyedflib is not installed correctly, run the following command:
```bash
conda install -c conda-forge pyedflib
```

## Usage
To create, train and test a new model, open ```ApneaDetection.py```. 

> **In ```ApneaDetection.py```**:

Select the polisomnography studies by specifying the file numbers. 

If datasets are not created yet, set ```path_edf``` and ```path_annot``` to the path where the ```.edf``` and ```.xml``` files are located, respectively, and run the following line: 
```python
ApneaDataset.create_datasets(files, path_edf, path_annot) 
```

Each file will be saved in a different dataset as a ```.pth``` file in ```.\data\ApneaDetection_HomePAPSignals\datasets``` folder. Each dataset will be saved split into 10 subsets.

For each file:

Line 42 will load the dataset
```python
dataset = ApneaDataset.load_dataset(f"./data/ApneaDetection_HomePAPSignals/datasets/dataset2_archivo_1600{file:03d}.pth")
```

The following lines will define the training, validation and testing subset indices for each fold:
```python
    train_subsets = [(fold + i) % 10 for i in range(8)]
    val_subsets = [(fold - 2 + i) % 10 for i in range(1)]
    test_subsets = [(fold - 1 + i) % 10 for i in range(1)]
```

The dataset will be undersampled to achieve a balanced class distribution.

Each fold will be run 5 times to get mean and std confusion matrix and metrics.

Each run will create and train a model, saving the best model (the one with less validation loss):
```python
model = Model(input_size, name2)
            model_arch = model.get_architecture()
            trainer = Trainer(
                model = model,
                trainset = trainset,
                valset = valset,
                n_epochs = 100,
                batch_size = 32,
                loss_fn = 'BCE',
                optimizer = 'SGD',
                lr = 0.01,
                momentum = 0,
                text = txt_file + data_analysis + model_arch,
                device = device)
            trainer.train(models_path + '/' + name0 + '/' + name1 + '/' + name2, verbose = True, plot = False, save_best_model = True)
```
Each run will test the best and final model, saving both confusion matrices and metrics. Then, the mean and std for all 5 runs will be calculated.
This repeats for 10 folds and the mean and std for all will be calculated.


> **In ```PlotSignals.py```**:

Select the polisomnography study by specifying the file number. For example, if file names are ```homepap-lab-full-1600001.edf``` and ```homepap-lab-full-1600001-profusion.xml``` then ```file = 1```. 
Set ```path_edf``` and ```path_annot``` to the path where the ```.edf``` and ```.xml``` files are located, respectivly, and run the following lines:
```python
all_signals = read_signals_EDF(path_edf)
annotations = get_annotations(path_annot)

bipolar_signal, tiempo, sampling = get_bipolar_signal(all_signals['C3'], all_signals['O1'])
plot_signals(annotations, tiempo, all_signals['C3']['Signal'], 'C3', all_signals['O1']['Signal'], 'O1', bipolar_signal, 'C3-O1')
```
![image](https://github.com/user-attachments/assets/2fc652f4-fb7a-4217-ba57-85eab6629e38)
