import os
import sys
import mne
from sklearn.model_selection import KFold, GroupKFold
import pickle
import torch
from torch.utils.data import TensorDataset
import pandas as pd
from model import CNNModel
from RealSignals import *
from Annotations import get_annotations

from Testing import Tester
from Training import Trainer
from collections import Counter
import random


def preprocess_signal(eeg_signal, sampling_freq, target_freq=128, verbose=False):
    """
    Applies preprocessing steps to the EEG signal.

    Args:
        eeg_signal (np.ndarray): The EEG signal.
        sampling_freq (float): The original sampling frequency.
        target_freq (float): The target sampling frequency after downsampling.

    Returns:
        np.ndarray: The preprocessed EEG signal.
    """
    # 1. Lowpass filter 
    eeg_signal = mne.filter.filter_data(eeg_signal, sampling_freq, l_freq=None, h_freq=45, verbose=verbose)

    # 2. Resample to 128 Hz
    eeg_signal = mne.filter.resample(eeg_signal, down=sampling_freq / target_freq)

    # 3. Z-score normalization
    eeg_signal = (eeg_signal - np.mean(eeg_signal)) / np.std(eeg_signal)

    return eeg_signal

def segment_and_label(eeg_signal, annotations, sampling_freq, segment_duration=30, apnea_threshold=10, ds='Shhs'):
    """
    Divides the data into 30-second segments and labels them based on the annotations.

    Args:
        eeg_signal (np.ndarray): The EEG signal.
        annotations (list): List of annotations with time of occurrence, type, and duration.
        sampling_freq (float): The sampling frequency.
        segment_duration (int): Duration of each segment in seconds.
        apnea_threshold (int): Minimum duration of apnea events in seconds to label a segment as apnea.

    Returns:
        list: List of segments.
        list: List of labels for each segment.
    """
    segment_length = int(segment_duration * sampling_freq)
    apnea_threshold_samples = int(apnea_threshold * sampling_freq)

    segments = []
    labels = []

    total_samples = len(eeg_signal)
    total_segments = total_samples // segment_length

    for i in range(total_segments):
        start_index = i * segment_length
        end_index = start_index + segment_length
        segment = eeg_signal[start_index:end_index]
        if len(segment) != segment_length:
            continue

        # Check if the segment contains at least 10 continuous seconds of OSA, MA, or hypopnea
        apnea_duration = 0
        for annotation in annotations:
            for anot in annotations[annotation]:
                annotation_start = anot[0]
                annotation_end = annotation_start + anot[1]
                if annotation_start >= start_index / sampling_freq and annotation_end <= end_index / sampling_freq:
                    apnea_duration += (annotation_end - annotation_start)
                elif annotation_start <= start_index / sampling_freq and annotation_end >= end_index / sampling_freq:
                    apnea_duration += (end_index / sampling_freq - start_index / sampling_freq)
                elif annotation_start < start_index / sampling_freq < annotation_end:
                    apnea_duration += (annotation_end - start_index / sampling_freq)
                elif annotation_start < end_index / sampling_freq < annotation_end:
                    apnea_duration += (end_index / sampling_freq - annotation_start)
                    

        label = 1 if apnea_duration >= apnea_threshold else 0
        segments.append(segment)
        labels.append(label)

    return segments, labels

def undersample_dataset(indices, labels):
    """
    Realiza undersampling en el conjunto de entrenamiento para balancear las clases.

    Args:
        indices (list): Lista de índices del conjunto de entrenamiento.
        labels (torch.Tensor): Tensor con las etiquetas correspondientes.

    Returns:
        list: Lista de índices balanceados después del undersampling.
    """
    label_counts = Counter([labels[i].item() for i in indices])
    majority_class = 0 if label_counts[0] > label_counts[1] else 1
    minority_class = 1 - majority_class

    majority_indices = [i for i in indices if labels[i].item() == majority_class]
    minority_indices = [i for i in indices if labels[i].item() == minority_class]

    random.seed(42) 
    majority_indices = random.sample(majority_indices, len(minority_indices))

    balanced_indices = majority_indices + minority_indices
    random.shuffle(balanced_indices)

    return balanced_indices



random.seed(42) #######

verbose = False
path_annot = 'C:\\Users\\elena\\OneDrive\\Documentos\\TFG\\Dataset\\shhs\\polysomnography\\annotations-events-profusion\\shhs2'
path_edf = 'C:\\Users\\elena\\OneDrive\\Documentos\\TFG\\Dataset\\shhs\\polysomnography\\edfs\\shhs2'
files = [f for f in os.listdir(path_edf) if f.endswith('.edf')]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models_path = '.\\models'
name0 = f'model_shhs_prueba'   ######CAMBIAR
if not os.path.exists(models_path + '\\' + name0):
    os.makedirs(models_path + '\\' + name0)

all_segments = []
all_labels = []
all_subjects = []

for file in files[:4]: ####BORRAR 
    file_path = os.path.join(path_edf, file)
    all_signals = read_signals_EDF(file_path)
    annotation_path = os.path.join(path_annot, file).replace('.edf', '-profusion.xml')
    annotations = get_annotations(annotation_path)

    eeg_signal, sampling_freq = all_signals['EEG']['Signal'], all_signals['EEG']['SamplingRate'] ##C4-A1: EEG

    eeg_signal = preprocess_signal(eeg_signal, sampling_freq, verbose=verbose)
    segments, labels = segment_and_label(eeg_signal, annotations, 128)

    subject_indices = list(range(len(labels)))
    balanced_indices = undersample_dataset(subject_indices, torch.tensor(labels))

    all_segments.extend([segments[i] for i in balanced_indices])
    all_labels.extend([labels[i] for i in balanced_indices])
    all_subjects.extend([file] * len(balanced_indices))

all_segments = torch.tensor(all_segments, dtype=torch.float32).unsqueeze(1)
all_labels = torch.tensor(all_labels, dtype=torch.int64)

########
unique_subjects = list(sorted(set(all_subjects)))
subject_to_idx = {suj: idx for idx, suj in enumerate(unique_subjects)}
subject_indices = [subject_to_idx[s] for s in all_subjects]
########

dataset = TensorDataset(all_segments, all_labels)
group_kfold = GroupKFold(n_splits=3)   #####CAMBIAR

for fold, (train_index, test_index) in enumerate(group_kfold.split(all_segments, all_labels, groups=subject_indices)): #########
    
    os.makedirs(models_path + '\\' + name0 + '\\' + name0 + f'_fold{fold}', exist_ok=True)
    
    balanced_train_dataset = Subset(dataset, train_index)
    train_loader = DataLoader(balanced_train_dataset, batch_size=32, shuffle=True)
    
    test_dataset = Subset(dataset, test_index)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    balanced_train_labels = [all_labels[i].item() for i in train_index]
    train_counts = Counter(balanced_train_labels)
    test_labels = [all_labels[i].item() for i in test_index]
    test_counts = Counter(test_labels)
    
    train_subjects = set([all_subjects[i] for i in train_index])
    test_subjects = set([all_subjects[i] for i in test_index])
    
    fold_info_path = os.path.join(models_path + '\\' + name0 + '\\' + name0 + f'_fold{fold}', f"fold_{fold}_info.txt")
    with open(fold_info_path, "w") as f:
        f.write(f"Fold {fold}:\n")
        f.write(f"Train segments after undersampling - No apnea: {train_counts[0]}, Apnea: {train_counts[1]}\n")
        f.write(f"Test segments - No apnea: {test_counts[0]}, Apnea: {test_counts[1]}\n")
        f.write(f"Train subjects: {', '.join(train_subjects)}\n")
        f.write(f"Test subjects: {', '.join(test_subjects)}\n")
    
    input_size = all_segments.shape[2]
    model = CNNModel(
            input_size = input_size
        ).to(device)
    
    trainer = Trainer(
        model = model,
        train_loader = train_loader,
        test_loader = test_loader,
        n_epochs = 2, #####CAMBIAR
        batch_size = 32,
        loss_fn = 'CE',
        optimizer = 'Adam',
        lr = 0.00163,
        momentum = 0,
        text = "",
        device = device)
    trainer.train(models_path + '\\' + name0 + '\\' + name0 + f'_fold{fold}', verbose = True, plot = False, save_model=True, save_best_model = False)
    
    tester = Tester(model = model,
                    test_loader = test_loader,
                    batch_size = 32,
                    device = device, 
                    best_final = 'final')
    cm, metrics = tester.evaluate(models_path + '\\' + name0 + '\\' + name0 + f'_fold{fold}', plot = False)
    
##############