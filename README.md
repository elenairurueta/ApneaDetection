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
Create environment:
```bash
conda create -n ApneaDetection python==3.11.9
```
Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage
To create, train and test a new model, open ```ApneaDetection.py```. 

> **In ```ApneaDetection.py```**:

Select the polisomnography study by specifying the file number. For example, if file names are ```homepap-lab-full-1600001.edf``` and ```homepap-lab-full-1600001-profusion.xml``` then ```file = 1```. 

If datasets are not created yet, set ```path_edf``` and ```path_annot``` to the path where the ```.edf``` and ```.xml``` files are located, respectively, and run the following line: 
```python
ApneaDataset.create_datasets([archivo], path_edf, path_annot)
```
The dataset will be saved as a ```.pth``` file in ```.\data\ApneaDetection_HomePAPSignals\datasets``` folder. The dataset will be saved split between train (80%), test (10%) and val (10%) subsets and having undersampled the majority class (in this case: not-apnea labelled as 0).

Line 16 will load the dataset:
```python
ApneaDataset.create_datasets([archivo], path_edf, path_annot)
```

The following lines will create and train the Model:
```python
model = Model(input_size, name)
model_arch = model.get_architecture()
trainer = Trainer(
    model = model, 
    trainset = dataset.trainset, 
    valset = dataset.valset, 
    n_epochs = 100, 
    batch_size = 32, 
    loss_fn = 'BCE',
    optimizer = 'SGD',
    lr = 0.01, 
    momentum = 0, 
    text = file_txt + data_analysis + model_arch)
trainer.train(models_path, verbose = False, plot = False, save_model = True, save_best_model = True)
```

And test it:
```python
tester = Tester(model = model, 
                testset = dataset.testset, 
                batch_size = 32)
tester.evaluate(models_path, plot = False)
```

To plot EEG derivations and a bipolar signal, open ```PlotSignals.py```.

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



