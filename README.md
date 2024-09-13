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
> **In ```GridSearch.py```**:
A grid search is implemented to explore different combinations of convolutional kernel sizes and dropout rates. The search is done across multiple folds for cross-validation. Hyperparameter search results are saved in hyperparameter_search_results.xlsx.


> **In ```ApneaDetection.py```**:

Select the polisomnography studies by specifying the file numbers. 

If datasets are not created yet, set ```path_edf``` and ```path_annot``` to the path where the ```.edf``` and ```.xml``` files are located, respectively, and run the following line: 
```python
ApneaDataset.create_datasets(files, path_edf, path_annot) 
```

Each file will be saved in a different dataset as a ```.pth``` file in ```.\data\ApneaDetection_HomePAPSignals\datasets``` folder. Each dataset will be saved split into 10 subsets.

The training process uses 10-fold cross-validation, where in each fold: 8 subsets per file are used for training, 1 for validation and 1 for testing.

All 12 datasets are combined into a single dataset. The combined dataset is then split into training, validation, and test sets based on the fold indices in each dataset. 
The training and validation subsets are undersampled to achieve a balanced class distribution.

The model is trained using the combined dataset.

After training, the model is evaluated in two ways:
* On the combined test dataset from all files
* On each file individually 
generating confusion matrices and evaluation metrics for each.

Also the best model from each fold is saved and tested on the joined testset and individual files for final evaluation.



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
