import os
from DataFormatting import ApneaDataset
from Models import Model
from Training import Trainer
from Testing import Tester
from sklearn.model_selection import KFold
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def undersample_segments(segments, labels, random_state=None):
    
    # Separate segments by class
    apnea_segments = np.array([seg for seg, label in zip(segments, labels) if label == 1])
    non_apnea_segments = np.array([seg for seg, label in zip(segments, labels) if label == 0])
    
    # Determine the minimum number of segments between the two classes
    min_count = min(len(apnea_segments), len(non_apnea_segments))
    
    if random_state is not None:
        np.random.seed(random_state)

    # Randomly select segments from each class to balance the dataset
    apnea_indices = np.random.choice(len(apnea_segments), min_count, replace=False)
    non_apnea_indices = np.random.choice(len(non_apnea_segments), min_count, replace=False)
    
    balanced_segments = np.concatenate((apnea_segments[apnea_indices], non_apnea_segments[non_apnea_indices]))
    balanced_labels = np.concatenate((np.ones(min_count), np.zeros(min_count))).reshape(-1, 1)
    
    return balanced_segments, balanced_labels


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

path_annot = "C:/Users/elena/OneDrive/Documentos/TFG/Dataset/HomePAP/polysomnography/annotations-events-profusion/lab/full"
path_edf = "C:/Users/elena/OneDrive/Documentos/TFG/Dataset/HomePAP/polysomnography/edfs/lab/full"

models_path = './models'
name0 = f'model_allfiles'
if not os.path.exists(models_path + '//' + name0):
    os.makedirs(models_path + '//' + name0)

#files = [int(f.split('_')[2]) for f in os.listdir("..//data//ApneaDetection_HomePAPSignals//datasets") if f.endswith('_C3O1.pth')]
excluded_files = {1600004, 1600043, 1600063, 1600072, 1600084, 1600095, 1600053, 1600055, 1600105, 1600113, 1600122, 1600151}
files = [int(f.split('-')[3].split('.')[0]) for f in os.listdir(path_edf) if f.endswith('.edf') and int(f.split('-')[3].split('.')[0]) not in excluded_files]

ApneaDataset.create_datasets(files, path_edf, path_annot, 
                             overlap = 10, perc_apnea = 0.3, 
                             signal=['C4', 'M1'],
                             filtering=True, filter = "FIR_Notch", 
                             split_dataset=False)

datasets = []

output_file = models_path + '//' + name0 + '//' + 'output.txt'
with open(output_file, 'w') as f:
    f.write(f"Number of files: {len(files)}\n")

    for file in files:
        try:
            ds = ApneaDataset.load_dataset(f"..//data//ApneaDetection_HomePAPSignals//datasets//dataset_file_{file}_C3O1.pth")
        except:
            f.write(f"Error loading dataset {file}")
            continue
        if(ds._ApneaDataset__sr != 200):
            ds.resample_segments(200)
        
        # Count apnea and non-apnea segments
        labels = ds[:][1]
        apnea_count = (labels == 1).sum().item()
        non_apnea_count = (labels == 0).sum().item()
        
        f.write(f"File {file}: {apnea_count} apnea segments, {non_apnea_count} non-apnea segments")
        datasets.append((file, ds))

    # Split subjects into 10 folds
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    folds = list(kf.split(datasets))

    for i, (train_index, test_index) in enumerate(folds):
        train_segments = []
        train_labels = []
        test_segments = []
        test_labels = []
        
        for idx in train_index:
            segments, labels = datasets[idx][1][:]
            # Ensure all segments have the same length
            if len(segments) == 0:
                continue
            segment_length = segments[0].shape[0]
            valid_segments = [seg for seg in segments if seg.shape[0] == segment_length]
            valid_labels = [labels[j] for j, seg in enumerate(segments) if seg.shape[0] == segment_length]
            train_segments.extend(valid_segments)
            train_labels.extend(valid_labels)
        
        for idx in test_index:
            segments, labels = datasets[idx][1][:]
            # Ensure all segments have the same length
            if len(segments) == 0:
                continue
            segment_length = segments[0].shape[0]
            valid_segments = [seg for seg in segments if seg.shape[0] == segment_length]
            valid_labels = [labels[j] for j, seg in enumerate(segments) if seg.shape[0] == segment_length]
            test_segments.extend(valid_segments)
            test_labels.extend(valid_labels)
        
        # Ensure all segments have the same shape
        train_segments = np.array([np.array(seg) for seg in train_segments])
        train_labels = np.array([np.array(lbl) for lbl in train_labels])
        test_segments = np.array([np.array(seg) for seg in test_segments])
        test_labels = np.array([np.array(lbl) for lbl in test_labels])
        
        train_segments, train_labels = undersample_segments(train_segments, train_labels, random_state=42)

        train_segments = torch.tensor(train_segments, dtype=torch.float32).unsqueeze(1)
        train_labels = torch.tensor(train_labels, dtype=torch.float32)
        test_segments = torch.tensor(test_segments, dtype=torch.float32).unsqueeze(1)
        test_labels = torch.tensor(test_labels, dtype=torch.float32)
        
        train_apnea_count = (train_labels == 1).sum().item()
        train_non_apnea_count = (train_labels == 0).sum().item()
        test_apnea_count = (test_labels == 1).sum().item()
        test_non_apnea_count = (test_labels == 0).sum().item()
        
        f.write(f"Fold {i}:\n")
        f.write(f"  Train subjects: {[datasets[idx][0] for idx in train_index]}\n")
        f.write(f"  Test subjects: {[datasets[idx][0] for idx in test_index]}\n")
        f.write(f"  Train - {train_apnea_count} apnea, {train_non_apnea_count} non-apnea\n")
        f.write(f"  Test - {test_apnea_count} apnea, {test_non_apnea_count} non-apnea\n")

        train_dataset = TensorDataset(train_segments, train_labels)
        test_dataset = TensorDataset(test_segments, test_labels)

        name1 = name0 + f'_fold{i}'
        if not os.path.exists(models_path + '//' + name0 + '//' + name1):
            os.makedirs(models_path + '//' + name0 + '//' + name1)
        
        input_size = train_segments.shape[2]

        for run in range(5):  # Run each fold 5 times
            name2 = name1 + f'_run{run}'
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
            
            trainer = Trainer(
                model = model,
                trainset = train_dataset,
                valset = test_dataset,
                n_epochs = 1,  # Adjust epochs as needed
                batch_size = 32,
                loss_fn = 'BCE',
                optimizer = 'SGD',
                lr = 0.01,
                momentum = 0,
                text = "",
                device = device)
            trainer.train(models_path + '//' + name0 + '//' + name1, verbose = True, plot = False, save_best_model = False)
            
            tester = Tester(model = model,
                            testset = test_dataset,
                            batch_size = 32,
                            device = device, 
                            best_final = 'final')
            cm, metrics = tester.evaluate(models_path + '//' + name0 + '//' + name1, plot = False)
            
            f.write(f"Fold {i}, Run {run} results:\n")
            f.write(f"  Confusion Matrix: {cm}\n")
            f.write(f"  Metrics: {metrics}\n")




