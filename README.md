# ApneaDetection
## Overview
This project is the final assignment for my Biomedical Engineering degree. It involves developing a machine learning model based on Convolutional Neural Networks (CNN) to detect sleep apnea in EEG signals.
## Initial Version
In the initial version of this project, simulated signals with characteristics similar to real EEG signals were used to test the model.
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

Download folder ApneaDetection_SimulatedSignals from: https://1drv.ms/f/s!Akd0CZdYW6D5gtNBfn6auvC5otEYLQ?e=KxyBGX

Password: ApneaDetection

Move folder to Data directory.
The path should look like: ```'Data\ApneaDetection_SimulatedSignals\SenalesCONapnea.csv'``` and ```'Data\ApneaDetection_SimulatedSignals\SenalesSINapnea.csv'```

## Usage
To create, train and test a new model, open ```NewModel.py```. To upload and test existing model, open ```ExistingModel.py```.

Lines 1 to 5 will import the required modules. 
```python
from LecturaSenalesSimuladas import *
from Modelo import *
from Training import *
from Testing import *
```
Line 8 will create a custom Dataset (ApneaDataset) with the simulated signals from the Data directory.
```python
dataset = ApneaDataset('Data\ApneaDetection_SimulatedSignals\SenalesCONapnea.csv', 'Data\ApneaDetection_SimulatedSignals\SenalesSINapnea.csv')
```

Line 13 will split the Dataset into train, validation and test Subsets. 
```python
dataset.split_dataset(train_perc = 0.6, 
                      val_perc = 0.2, 
                      test_perc = 0.2)
```
You can modify the following hiperparameters:
- ```train_perc``` (float): percentage of training data (0 < train_perc < 1)
- ```val_perc``` (float): percentage of validation data (0 < val_perc < 1)
- ```test_perc``` (float): percentage of test data (0 < test_perc < 1)

```train_perc``` + ```val_perc``` + ```test_perc``` must be = 1, otherwise ```test_perc``` will be automatically calculated to ensure this condition is satisfied.

> **In ```NewModel.py```**:

Lines 20 to 36 will create and train the new model.
  ```pyhton
  nombre = 'modelo'
  model = Model(input_size, nombre)
  trainer = Trainer(
      model = model, 
      trainset = dataset.trainset, 
      valset = dataset.valset, 
      n_epochs = 100, 
      batch_size = 32, 
      loss_fn = 'BCE',
      optimizer = 'SGD',
      lr = 0.01, 
      momentum = 0.5,
      text = '')
  trainer.train(verbose = True, plot = True)
  ```
  You can modify the following hiperparameters:
  - ```nombre``` (string): name used to save the model and figures
  - ```n_epochs``` (int): number of epochs to train the model
  - ```batch_size``` (int): number of data used in one iteration
  - ```lr``` (float): learning rate
  - ```momentum``` (float): (0 <= momentum <= 1)
  - ```verbose``` (bool): if verbose = False it will not print to console but will be saved in the ```models\nombre\nombre_training.txt``` file
  - ```plot``` (bool): if plot = False the figures will not be displayed but will be saved in the ```models\nombre\nombre_''.png``` files
  
  In this first version, only 'BCE' loss function and 'SGD' optimizer are available.
  
  Lines 38 to 41 will test the new model. 
  ```pyhton
  tester = Tester(model = model, 
                  testset = dataset.testset, 
                  batch_size = 32)
  tester.evaluate(plot = True)
  ```
  You can modify the following hiperparameters:
  - ```batch_size``` (int): number of data used in one iteration
  
  Line 43 will save the new model in ```'models\nombre\nombre.pt/pth'``` directory.

> **In ```ExistingModel.py```**:

Lines 20 to 28 will upload and test an existing model. 
  ```pyhton
  nombre = 'modelo'
  model = Model.load_model(nombre, input_size)
  tester = Tester(model = model, 
                  testset = dataset.testset, 
                  batch_size = 32)
  tester.evaluate(plot = True)
  ```
  You can modify the following hiperparameters:
  - ```nombre``` (string): name used to upload the model. The path should look like: ```'models\nombre\nombre.pt/pth'```. The file extension should be ```'.pt'``` or ```'.pth'```. 
  - ```batch_size``` (int): number of data used in one iteration
  - ```plot``` (bool): if plot = False the figures will not be displayed but will be saved in the ```models\nombre\nombre_''.png``` files

