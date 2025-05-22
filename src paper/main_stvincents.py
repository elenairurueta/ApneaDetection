import os
import sys
import mne
import numpy as np
import torch
from datetime import datetime
from torch.utils.data import TensorDataset, DataLoader, Subset
from sklearn.model_selection import KFold
from model import CNNModel
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src CNN')))
from Training import Trainer
from Testing import Tester

def read_eeg_signal(file_path, channel_name='C4A1'):
    """
    Reads the EEG signal from the specified file and channel.

    Args:
        file_path (str): Path to the EDF file.
        channel_name (str): Name of the EEG channel to read.

    Returns:
        np.ndarray: The EEG signal.
        float: The sampling frequency.
    """
    raw = mne.io.read_raw_edf(file_path, preload=True)
    raw.pick_channels([channel_name])
    eeg_signal = raw.get_data()[0]
    sampling_freq = raw.info['sfreq']
    return eeg_signal, sampling_freq

def read_annotations(file_path):
    """
    Reads the annotations from the specified file.

    Args:
        file_path (str): Path to the annotation file.

    Returns:
        list: List of annotations with time of occurrence, type, and duration.
    """
    annotations = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines[2:]:  # Skip the first two lines
            parts = line.strip().split()
            if len(parts) >= 4:  # Ensure there are enough parts to extract the necessary information
                time_of_occurrence = parts[0]
                event_type = parts[1]
                try:
                    duration = float(parts[3])  # Attempt to convert the duration to a float
                except ValueError:
                    continue  # Skip this line if the duration is not a valid float
                annotations.append((time_of_occurrence, event_type, duration))
    return annotations
    

def preprocess_signal(eeg_signal, sampling_freq, target_freq=125):
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
    eeg_signal = mne.filter.filter_data(eeg_signal, sampling_freq, l_freq=None, h_freq=45)

    # 2. Downsample (125 Hz)
    eeg_signal = mne.filter.resample(eeg_signal, down=round(sampling_freq / target_freq))

    # 3. Z-score normalization
    eeg_signal = (eeg_signal - np.mean(eeg_signal)) / np.std(eeg_signal)

    return eeg_signal

def segment_and_label(eeg_signal, annotations, sampling_freq, segment_duration=30, apnea_threshold=10):
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

    for start in range(0, len(eeg_signal), segment_length):
        end = start + segment_length
        segment = eeg_signal[start:end]
        if len(segment) < segment_length:
            break

        # Check if the segment contains at least 10 continuous seconds of OSA, MA, or hypopnea
        apnea_duration = 0
        for annotation in annotations:
            time_str = annotation[0]
            time_obj = datetime.strptime(time_str, '%H:%M:%S')
            annotation_start = int((time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second) * sampling_freq)
            annotation_end = annotation_start + int(annotation[2] * sampling_freq)
            if annotation[1] in ['O', 'M', 'HYP']:
                if annotation_start >= start and annotation_end <= end:
                    apnea_duration += (annotation_end - annotation_start)
                elif annotation_start < start < annotation_end:
                    apnea_duration += (annotation_end - start)
                elif annotation_start < end < annotation_end:
                    apnea_duration += (end - annotation_start)

        label = 1 if apnea_duration >= apnea_threshold_samples else 0
        segments.append(segment)
        labels.append(label)

    return segments, labels

def main():
    dataset_path = "C:\\Users\\elena\\OneDrive\\Documentos\\TFG\\Dataset\\st-vincents-university-hospital-university-college-dublin-sleep-apnea-database-1.0.0\\files"
    files = [f for f in os.listdir(dataset_path) if f.endswith('.edf')]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_path = './models'
    name0 = 'model_paper'
    if not os.path.exists(models_path + '/' + name0):
        os.makedirs(models_path + '/' + name0)

    all_segments = []
    all_labels = []

    for file in files:
        file_path = os.path.join(dataset_path, file)
        annotation_file = file_path.replace('.edf', '_respevt.txt')
        
        # get signal (.rec files St. Vincent's University Hospital / University College Dublin Sleep Apnea Database)
        # we used channel C4-A1 
        # get annotations (_respevt.txt)
        eeg_signal, sampling_freq = read_eeg_signal(file_path)
        annotations = read_annotations(annotation_file)
        
        # Apply preprocessing steps
        eeg_signal = preprocess_signal(eeg_signal, sampling_freq)
        
        # Segment and label the data
        segments, labels = segment_and_label(eeg_signal, annotations, sampling_freq)
        
        all_segments.extend(segments)
        all_labels.extend(labels)

    # Convert segments and labels to tensors
    all_segments = torch.tensor(all_segments, dtype=torch.float32).unsqueeze(1)
    all_labels = torch.tensor(all_labels, dtype=torch.float32)
    
    # Create dataset
    dataset = TensorDataset(all_segments, all_labels)

    # Perform 10-fold cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    fold = 0
    for train_index, test_index in kf.split(dataset):
        train_dataset = Subset(dataset, train_index)
        test_dataset = Subset(dataset, test_index)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        input_size = all_segments.shape[2]
        model = CNNModel(
                input_size = input_size
            ).to(device)
        
        # 10-fold cross-validation
        # Network training was performed for 40 epochs
        # Cross-entropy loss function. 
        # optimized using Adam: alpha coefficient was set at 0.9, and beta coefficient at 0.999. 
        # The learning rate was tuned alongside other hyperparameters. lr = 0.00163
        trainer = Trainer(
            model = model,
            trainset = train_loader,
            valset = test_loader,
            n_epochs = 40,
            batch_size = 32,
            loss_fn = 'CE',
            optimizer = 'Adam',
            lr = 0.00163,
            momentum = 0,
            text = "",
            device = device)
        trainer.train(models_path + '/' + name0 + f'_fold{fold}', verbose = True, plot = False, save_best_model = True)
        
        tester = Tester(model = model,
                        testset = test_loader,
                        batch_size = 32,
                        device = device, 
                        best_final = 'final')
        cm, metrics = tester.evaluate(models_path + '/' + name0 + f'_fold{fold}', plot = False)
        
        print(f"Fold {fold} results:")
        print(f"Confusion Matrix: {cm}")
        print(f"Metrics: {metrics}")
        
        fold += 1

if __name__ == "__main__":
    main()



# To quantify performance of our trained CNN, we used accuracy and Matthews correlation coefficient (MCC).
# We used a Bayesian t-test, computing 95% highest density intervals (HDIs), to compare our CNN’s performance to baseline.