from pyedflib import EdfReader
import xml.etree.ElementTree as ET
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import butter, filtfilt
import seaborn as sns
import pandas as pd
import random
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import os
import pickle 

def main_single_file():
    file = 151

    path_annot = "C:/Users/elena/OneDrive/Documentos/TFG/Dataset/HomePAP/polysomnography/annotations-events-profusion/lab/full"
    path_edf = "C:/Users/elena/OneDrive/Documentos/TFG/Dataset/HomePAP/polysomnography/edfs/lab/full"

    all_signals = read_signals_EDF(path_edf + f"\\homepap-lab-full-1600{str(file).zfill(3)}.edf")
    annotations = get_annotations(path_annot + f"\\homepap-lab-full-1600{str(file).zfill(3)}-profusion.xml")

    """Uso señales bipolares C3-M2(C3-A2) y C4-M1(C4-A1)"""
    bipolar_signal1, time1, sampling1 = get_bipolar_signal(all_signals['C3'], all_signals['M2'])
    bipolar_signal2, time2, sampling2 = get_bipolar_signal(all_signals['C4'], all_signals['M1'])
    """AVERAGING IN TIME DOMAIN"""
    avg_signal = (bipolar_signal1 + bipolar_signal2) / 2

    """PLOT C3-M2, C4-M1 AND AVERAGE"""
    # plt.figure(figsize=(12, 6))
    # plt.plot(time2, bipolar_signal2, label="Bipolar C4-A1", alpha=0.7)
    # plt.plot(time1, bipolar_signal1, label="Bipolar C3-A2", alpha=0.7)
    # plt.plot(time1, avg_signal, label="Señal Promedio", linestyle="dashed", linewidth=2, color="black")
    # plt.xlabel("Tiempo (s)")
    # plt.ylabel("Amplitud")
    # plt.title("Señales Bipolares y Promedio")
    # plt.legend()
    # plt.grid()
    # plt.show()

    """Segmentos de la señal promediada con longitud de 30seg, overlap de 10seg y 30% de apnea, descartando 5min"""
    segments_EEG = get_signal_segments(avg_signal, time1, sampling1, annotations, period_length=30, overlap=10, perc_apnea=0.3, t_descarte = 5*60)
    label_counts = Counter(segment['Label'] for segment in segments_EEG)
    print("Before undersampling, before filtering outliers")
    print(f"Cantidad de segmentos con Label = 0: {label_counts.get(0, 0)}")
    print(f"Cantidad de segmentos con Label = 1: {label_counts.get(1, 0)}")

    # """Undersampling de segmentos para que haya la misma cantidad con apnea que sin apnea"""
    # segments_EEG = undersample_segments(segments_EEG)
    # label_counts = Counter(segment['Label'] for segment in segments_EEG)
    # print("Undersampling")
    # print(f"Cantidad de segmentos con Label = 0: {label_counts.get(0, 0)}")
    # print(f"Cantidad de segmentos con Label = 1: {label_counts.get(1, 0)}")
    
    """PRE-PROCESSING: remove dc offset, amplitude normalisation"""
    segments_EEG = segment_preprocessing(segments_EEG, plot = False)
    
    bands = {
        "delta": (0.25, 4),
        "theta": (4, 8),
        "alpha": (8, 12),
        "sigma": (12, 16),
        "beta": (16, 40)
    }
    """ENERGIES COMPUTATION"""
    segments_EEG = compute_band_energy(segments_EEG, bands)
    segments_EEG = compute_energy_ratios(segments_EEG, bands)
    #segments_EEG = compute_energy_ratios(segments_EEG, bands, norm = True) no normalizar

    """energy is mainly shifted from lower- to higher-frequency bands"""
    selected_ratios = ['Ratio_delta_theta', 'Ratio_delta_alpha', 'Ratio_delta_sigma', 'Ratio_delta_beta', 'Ratio_theta_alpha']
    
    plot_energy_ratios_boxplot(segments_EEG, selected_ratios)

    segments_EEG_filtered,_,_ = detect_and_plot_outliers(segments_EEG, selected_ratios, bands, threshold_percentile=10, 
                                                        saveFigures=False, pathFigures = "C:\\dev\\ApneaDetection2\\src KNN\\outliers\\")
    
    label_counts = Counter(segment['Label'] for segment in segments_EEG_filtered)
    print("Before undersampling, after filtering outliers")
    print(f"Cantidad de segmentos con Label = 0: {label_counts.get(0, 0)}")
    print(f"Cantidad de segmentos con Label = 1: {label_counts.get(1, 0)}")

    segments_EEG_filtered = undersample_segments(segments_EEG_filtered)

    label_counts = Counter(segment['Label'] for segment in segments_EEG_filtered)
    print("After Undersampling")
    print(f"Cantidad de segmentos con Label = 0: {label_counts.get(0, 0)}")
    print(f"Cantidad de segmentos con Label = 1: {label_counts.get(1, 0)}")

    plot_energy_ratios_boxplot(segments_EEG_filtered, selected_ratios)

    features = []
    labels = []

    for segment in segments_EEG_filtered:
        #ratio_features = [segment[key] for key in segment if key.startswith('Ratio')]
        ratio_features = [segment[key] for key in selected_ratios]
        features.append(ratio_features)
        labels.append(segment['Label'])

    X = np.array(features)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    #k_values = [3, 5, 7, 9, 11, 13, 15]
    k_values = [5]
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        print(f'n_neighbors={k}, Accuracy={accuracy_score(y_test, y_pred)}')
        print(f'n_neighbors={k}, Precisión={precision_score(y_test, y_pred)}')
        print(f'n_neighbors={k}, Recall={recall_score(y_test, y_pred)}')
        print(f'n_neighbors={k}, F1-score={f1_score(y_test, y_pred)}')
        print(f'n_neighbors={k}, Cross-validated accuracy={cross_val_score(knn, X, y, cv=5, scoring="accuracy").mean()}')
        print(f'n_neighbors={k}, Cross-validated precision_macro={cross_val_score(knn, X, y, cv=5, scoring="precision_macro").mean()}')
        print(f'n_neighbors={k}, Cross-validated recall_macro={cross_val_score(knn, X, y, cv=5, scoring="recall_macro").mean()}')
        print(f'n_neighbors={k}, Cross-validated f1_macro={cross_val_score(knn, X, y, cv=5, scoring="f1_macro").mean()}')

        plot_confusion_matrix(y_test, y_pred)


def main_multiple_files():
    files = [4, 43, 53, 55, 63, 72, 84, 95, 105, 113, 122, 151]
    combined_segments = []
    for file in files:
        path_annot = "C:/Users/elena/OneDrive/Documentos/TFG/Dataset/HomePAP/polysomnography/annotations-events-profusion/lab/full"
        path_edf = "C:/Users/elena/OneDrive/Documentos/TFG/Dataset/HomePAP/polysomnography/edfs/lab/full"

        all_signals = read_signals_EDF(path_edf + f"\\homepap-lab-full-1600{str(file).zfill(3)}.edf")
        annotations = get_annotations(path_annot + f"\\homepap-lab-full-1600{str(file).zfill(3)}-profusion.xml")

        bipolar_signal1, time1, sampling1 = get_bipolar_signal(all_signals['C3'], all_signals['M2'])
        bipolar_signal2, time2, sampling2 = get_bipolar_signal(all_signals['C4'], all_signals['M1'])
        """AVERAGING IN TIME DOMAIN"""
        avg_signal = (bipolar_signal1 + bipolar_signal2) / 2

        segments_EEG = get_signal_segments(avg_signal, time1, sampling1, annotations, period_length=30, overlap=10, perc_apnea=0.3, t_descarte = 5*60)
        #segments_EEG = undersample_segments(segments_EEG)
        
        """PRE-PROCESSING: remove dc offset, amplitude normalisation"""
        segments_EEG = segment_preprocessing(segments_EEG, plot = False)
        combined_segments += segments_EEG
    
    """ENERGIES COMPUTATION"""
    bands = {
            "delta": (0.25, 4),
            "theta": (4, 8),
            "alpha": (8, 12),
            "sigma": (12, 16),
            "beta": (16, 40)
        }
    
    label_counts = Counter(segment['Label'] for segment in combined_segments)
    print(f"Cantidad de segmentos con Label = 0: {label_counts.get(0, 0)}")
    print(f"Cantidad de segmentos con Label = 1: {label_counts.get(1, 0)}")

    combined_segments = compute_band_energy(combined_segments, bands)
    combined_segments = compute_energy_ratios(combined_segments, bands)
    
    """energy is mainly shifted from lower- to higher-frequency bands"""
    selected_ratios = ['Ratio_delta_theta', 'Ratio_delta_alpha', 'Ratio_delta_sigma', 'Ratio_delta_beta', 'Ratio_theta_alpha']
    
    plot_energy_ratios_boxplot(combined_segments, selected_ratios)
    #plot_energy_ratio_variation(combined_segments, 'Ratio_delta_beta')

    #combined_segments = undersample_segments(combined_segments)

    segments_EEG_filtered,_,_ = detect_and_plot_outliers_2(combined_segments, selected_ratios, bands, threshold_percentile=10, 
                                                        saveFigures=False, pathFigures = "C:\\dev\\ApneaDetection2\\src KNN\\outliers\\")
    
    label_counts = Counter(segment['Label'] for segment in segments_EEG_filtered)
    print("Before undersampling, after filtering outliers")
    print(f"Cantidad de segmentos con Label = 0: {label_counts.get(0, 0)}")
    print(f"Cantidad de segmentos con Label = 1: {label_counts.get(1, 0)}")

    segments_EEG_filtered = undersample_segments(segments_EEG_filtered)

    label_counts = Counter(segment['Label'] for segment in segments_EEG_filtered)
    print("After Undersampling")
    print(f"Cantidad de segmentos con Label = 0: {label_counts.get(0, 0)}")
    print(f"Cantidad de segmentos con Label = 1: {label_counts.get(1, 0)}")

    plot_energy_ratios_boxplot(segments_EEG_filtered, selected_ratios)


    features = []
    labels = []

    for segment in segments_EEG_filtered:
        ratio_features = [segment[key] for key in selected_ratios]
        features.append(ratio_features)
        labels.append(segment['Label'])

    X = np.array(features)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    k_values = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21] ### k impar para evitar empates
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        print(f'n_neighbors={k}, Accuracy={accuracy_score(y_test, y_pred)}')
        print(f'n_neighbors={k}, Precisión={precision_score(y_test, y_pred)}')
        print(f'n_neighbors={k}, Recall={recall_score(y_test, y_pred)}')
        print(f'n_neighbors={k}, F1-score={f1_score(y_test, y_pred)}')
        print(f'n_neighbors={k}, Cross-validated accuracy={cross_val_score(knn, X, y, cv=5, scoring="accuracy").mean()}')
        print(f'n_neighbors={k}, Cross-validated precision_macro={cross_val_score(knn, X, y, cv=5, scoring="precision_macro").mean()}')
        print(f'n_neighbors={k}, Cross-validated recall_macro={cross_val_score(knn, X, y, cv=5, scoring="recall_macro").mean()}')
        print(f'n_neighbors={k}, Cross-validated f1_macro={cross_val_score(knn, X, y, cv=5, scoring="f1_macro").mean()}')

        plot_confusion_matrix(y_test, y_pred)


def read_signals_EDF(path:str):
    """
    Reads signals from an EDF file and stores them in a dictionary.

    Parameters:
    - path (str): The path to the EDF file.

    Returns:
    - all_signals: A dictionary where each key is a signal label and each value is a dictionary with the signal data, time points, dimension, and sampling rate.
    """

    EDF = EdfReader(path)
    labels = EDF.getSignalLabels()
    all_signals = {}
    for channel in range(0, EDF.signals_in_file):
        signal = EDF.readSignal(channel)
        time = np.arange(0, EDF.getFileDuration(), 1/EDF.getSampleFrequency(channel))
        all_signals[labels[channel]] = {'Signal': signal, 'Time': time, 'Dimension': EDF.getPhysicalDimension(channel), 'SamplingRate': EDF.getSampleFrequency(channel)}
    return all_signals

def get_annotations(path:str, eventos = ['Obstructive Apnea', 'Central Apnea', 'Mixed Apnea']):
    tree = ET.parse(path)
    root = tree.getroot()

    annotations = {}

    scored_events = root.find('ScoredEvents')
    for evento in eventos:
        evento_annotations = []
        for scored_event in scored_events.findall('ScoredEvent'):
            name = scored_event.find('Name').text
            if name == evento:
                start = float(scored_event.find('Start').text)
                duration = float(scored_event.find('Duration').text)
                evento_annotations.append([start, duration])
        annotations[evento] = evento_annotations

    return annotations

def get_bipolar_signal(signal1, signal2):
    """
    Computes the bipolar signal by subtracting two signals with the same sampling rate.

    Parameters:
    - signal1: A dictionary containing 'Signal' (the first signal array), 'Time' (the time points), and 'SamplingRate' (the sampling rate).
    - signal2: A dictionary containing 'Signal' (the second signal array), 'Time' (the time points), and 'SamplingRate' (the sampling rate).

    Returns:
    - signal: The bipolar signal (difference between signal1 and signal2).
    - time: The time points corresponding to the signal.
    - sampling: The sampling rate of the signal.
    """

    if(signal1['SamplingRate'] != signal2['SamplingRate']):
        raise TypeError("Signals not compatible: different sampling rate")
    signal = signal1['Signal'] - signal2['Signal']
    time = signal1['Time']
    sampling = signal1['SamplingRate']
    return signal, time, sampling

def get_signal_segments(signal, time, sampling_rate, annotations, period_length=30, overlap=10, perc_apnea=0.3, t_descarte = 0):
    """
    Divides the signal into overlapping segments and labels each one based on the percentage of apnea present during the segment.

    Parameters:
    - signal: The full signal array from which to extract segments.
    - time: Array of time points corresponding to the signal.
    - sampling_rate: The sampling rate of the signal (in Hz).
    - annotations: A dictionary where each key corresponds to an event type, and the values are lists of apnea events (start time, duration).
    - period_length: The length of each segment in seconds (default is 30 seconds).
    - overlap: The amount of overlap between consecutive segments in seconds (default is 10 seconds).
    - perc_apnea: The minimum percentage of an apnea event in a segment to classify the segment as containing apnea (default is 30%).

    Returns:
    - A list of dictionaries, each containing:
    - 'Signal': The extracted signal segment.
    - 'Label': 1 if apnea is present in the segment, 0 otherwise.
    - 'Start': The start time of the segment.
    - 'End': The end time of the segment.
    - 'SamplingRate': The sampling rate used to extract the segment.
    """

    step = period_length - overlap
    num_periods = int(np.ceil((time[-1] - 2*t_descarte - period_length) / step))+1 #Calculates the number of periods based on the total signal duration, segment length, and overlap.
    segments = []

    for i in range(num_periods):
        period_start = t_descarte + i * step
        period_end = period_start + period_length
        start_index = int(period_start * sampling_rate)
        end_index = int(period_end * sampling_rate)
        label = 0
        if end_index > len(signal) - int(t_descarte * sampling_rate):
            break

        for annotation in annotations:
            for anot in annotations[annotation]:
                #Checks if any apnea event overlaps with the segment.
                start, duration = anot
                apnea_start = start
                apnea_end = start + duration

                if (apnea_start < period_end and apnea_end > period_start):
                    order = [period_start, apnea_start, apnea_end, period_end]
                    order.sort()
                    apnea_in_segment = order[2] - order[1]
                    if(apnea_in_segment >= perc_apnea*duration):
                        label = 1
                    break
                #If the overlap of the apnea event in the segment exceeds the specified percentage, the segment is labeled as containing apnea (Label = 1); otherwise, it is labeled as not containing apnea (Label = 0)
        segments.append({'Signal': signal[start_index:end_index], 'Label': label, 'Start': period_start, 'End': period_end, 'SamplingRate': sampling_rate})
        if(i>0):
            if(len(segments[-1]['Signal']) < len(segments[0]['Signal'])):
                segments.pop()
    return segments

def get_signal_segments_strict(signal, tiempo, sampling_rate, annotations):
    
    all_annotations = []
    segments_withapnea = []
    segments_withoutapnea = []

    for event in annotations:
        all_annotations = all_annotations + annotations[event]
    all_annotations.sort(key=lambda x: x[0]) #Annotations sorted by start time

    for idx, annotation in enumerate(all_annotations):
            start_apnea, length_apnea = annotation
            end_apnea = start_apnea + length_apnea
            if(idx < len(all_annotations) - 1):
                inicio_prox = all_annotations[idx + 1][0]
                if (end_apnea + 15) <= inicio_prox:
                    #15 seconds before the end of the apnea and 15 seconds after ensuring there is not another apnea
                    start_time = end_apnea - 15
                    start_index = int(start_time * sampling_rate)
                    end_time = end_apnea + 15
                    end_index = int(end_time * sampling_rate)
                    segments_withapnea.append({'Signal': signal[start_index:end_index], 'Label': 1, 'Start': start_time, 'End': end_time, 'SamplingRate': sampling_rate})
                    if(len(segments_withapnea[-1]['Signal']) < len(segments_withapnea[0]['Signal'])):
                        segments_withapnea.pop()

    t = tiempo[0]
    while(t<tiempo[-1]):
        has_apnea = False
        for annotation in all_annotations:
            start_apnea, length_apnea = annotation
            end_apnea = start_apnea + length_apnea
            if(((start_apnea >= t) and (start_apnea <= t + 45)) or
               ((end_apnea >= t) and (end_apnea <= t + 45)) or
               ((start_apnea < t) and (end_apnea > t + 45))):
                #Ensuring there is no apnea in the previous 15 seconds nor in the 30-second segment
                has_apnea = True
                t = end_apnea + 1
                break
        if(has_apnea == False):
            start_index = int((t+15) * sampling_rate)
            end_index = int((t+45) * sampling_rate)
            segments_withoutapnea.append({'Signal': signal[start_index:end_index], 'Label': 0, 'Start': t+15, 'End': t+45, 'SamplingRate': sampling_rate})
            if(len(segments_withoutapnea[-1]['Signal']) < len(segments_withoutapnea[0]['Signal'])):
                segments_withoutapnea.pop()
            t = t + 30
    
    #Combined the apnea and non-apnea segments and shuffled them randomly
    segments = segments_withapnea + segments_withoutapnea
    segments.sort(key=lambda x: x['Start'])
    random.shuffle(segments)

    return segments

def undersample_segments(segments_EEG, seed=42):
    segments_0 = [seg for seg in segments_EEG if seg['Label'] == 0]
    segments_1 = [seg for seg in segments_EEG if seg['Label'] == 1]
    if len(segments_0) > len(segments_1):
        majority_segments, minority_segments = segments_0, segments_1
    else:
        majority_segments, minority_segments = segments_1, segments_0

    minority_count = len(minority_segments) 
    np.random.seed(seed)
    undersampled_majority = np.random.choice(majority_segments, size=minority_count, replace=False)
    balanced_segments = list(undersampled_majority) + minority_segments
    np.random.shuffle(balanced_segments)

    return balanced_segments

def segment_preprocessing(segments_EEG, plot = True):
    
    for segment in segments_EEG:
        """REMOVING DC OFFSET: subtracting the mean value of that frame from each sample value"""
        segmento_vmedio = segment['Signal'] - np.mean(segment['Signal'])
        """AMPLITUDE NORMALISATION: with reference to the maximum sample value of that frame"""
        segmento_norm = segmento_vmedio / np.max(np.abs(segmento_vmedio))

        if segmento_norm is None or np.isnan(segmento_norm).all():
            segmento_norm = segmento_vmedio

        segment['Signal'] = segmento_norm

        if(plot):
            plt.figure(figsize=(12, 6))
            tiempo = np.arange(0, 30, 1/segment['SamplingRate'])
            plt.plot(tiempo, segment['Signal'], label="1. Segmento promediado", alpha=0.5)
            plt.plot(tiempo, segmento_vmedio, label="2. Segmento - valor medio", alpha=0.5)
            plt.plot(tiempo, segmento_norm, label="3. Segmento normalizado", alpha=0.5)
            plt.xlabel("Tiempo (s)")
            plt.ylabel("Amplitud")
            plt.title(f"Segmento con label: {segment['Label']}")
            plt.legend()
            plt.grid()
            plt.show()

    return segments_EEG

def bandpass_filter(signal, fs, f_min, f_max, order=4):
    nyquist = fs / 2
    b, a = butter(order, [f_min / nyquist, f_max / nyquist], btype='band')
    return filtfilt(b, a, signal)

def compute_band_energy(segments_EEG, bands, saveFigures = False):
    
    processed_segments = []

    os.makedirs("segments_filtered", exist_ok=True)

    for idx, segment in enumerate(segments_EEG):
        signal = segment['Signal']
        fs = segment['SamplingRate']
        time = np.linspace(0, len(signal) / fs, len(signal))
        if saveFigures:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(time, signal, label="Señal Original", color="blue", alpha=0.6)
            
        band_energies = {}
        for band_name, (f_min, f_max) in bands.items():

            """BAND LIMITED SIGNAL EXTRACTION"""
            filtered_signal = bandpass_filter(signal, fs, f_min, f_max)
            if saveFigures:
                ax.plot(time, filtered_signal, label=band_name, alpha=0.6)

            """ENERGY IN TIME DOMAIN: xp[n]^2"""
            energy = np.sum(filtered_signal[100:-100] ** 2)
            band_energies[f'Energy_{band_name}'] = energy

        processed_segment = {**segment, **band_energies}
        processed_segments.append(processed_segment)
        if saveFigures:
            ax.set_xlabel("Tiempo (s)")
            ax.set_ylabel("Amplitud")
            ax.legend()
            ax.grid(True)
            filename = os.path.join("segments_filtered", f"segment_{idx + 1}.png")
            plt.savefig(filename, dpi=300)
            filename = os.path.join("segments_filtered", f"segment_{idx + 1}.pkl")
            with open(filename, "wb") as f:
                pickle.dump(fig, f)
            plt.close()

    return processed_segments

def compute_energy_ratios(segments_EEG, bands, norm = False):
    
    band_names = list(bands.keys())
    processed_segments = []
    epsilon = 1e-20 #para evitar la división por cero

    all_ratios = {f'Ratio_{band_p}_{band_q}': [] 
                  for i in range(len(band_names)) 
                  for j in range(i + 1, len(band_names)) 
                  for band_p, band_q in [(band_names[i], band_names[j]), (band_names[j], band_names[i])]} 

    for segment in segments_EEG:
        energy_ratios = {}

        for i in range(len(band_names)):
            for j in range(i + 1, len(band_names)):

                band_p, band_q = band_names[i], band_names[j]
                
                Ep = segment.get(f'Energy_{band_p}', 1e-10)
                Eq = segment.get(f'Energy_{band_q}', 1e-10)

                if np.isnan(Ep) or np.isnan(Eq):
                    raise ValueError(f"❌ Error: NaN detectado en {segment}\n"
                                     f"Energy_{band_p} = {Ep}, Energy_{band_q} = {Eq}")

                ratio_pq = Ep / (Eq + epsilon)
                ratio_qp = Eq / (Ep + epsilon)

                energy_ratios[f'Ratio_{band_p}_{band_q}'] = ratio_pq
                energy_ratios[f'Ratio_{band_q}_{band_p}'] = ratio_qp

                all_ratios[f'Ratio_{band_p}_{band_q}'].append(ratio_pq)
                all_ratios[f'Ratio_{band_q}_{band_p}'].append(ratio_qp)

        processed_segment = {**segment, **energy_ratios}
        processed_segments.append(processed_segment)

    if(norm):
        max_values = {key: max(values) if len(values) > 0 else 1 for key, values in all_ratios.items()}

        for segment in processed_segments:
            for key in max_values:
                if key in segment:
                    segment[key] /= max_values[key]

    return processed_segments

def detect_and_plot_outliers_1(segments_EEG, ratio_names, bands_list, threshold_percentile=2, saveFigures=False, pathFigures = "", deleteOutliers=True):
    
    outlier_info = {}  # Diccionario que almacena qué segmentos son outliers en qué ratios
    outlier_indices = set()  # Conjunto para guardar los índices únicos de outliers

    for ratio_name in ratio_names:
        ratio_values = [segment[ratio_name] for segment in segments_EEG if ratio_name in segment]
        
        if not ratio_values:
            print(f"No se encontraron valores para {ratio_name}.")
            continue  

        lower_threshold = np.percentile(ratio_values, threshold_percentile)
        upper_threshold = np.percentile(ratio_values, 100 - threshold_percentile)

        print(f"Thresholds para {ratio_name}: Inferior={lower_threshold:.4f}, Superior={upper_threshold:.4f}")

        for idx, segment in enumerate(segments_EEG):
            if ratio_name in segment and (segment[ratio_name] < lower_threshold or segment[ratio_name] > upper_threshold):
                outlier_indices.add(idx)  # Agregamos el índice del outlier
                if idx not in outlier_info:
                    outlier_info[idx] = []
                outlier_info[idx].append(ratio_name)

    # Lista de segmentos que son outliers
    outlier_segments = [segments_EEG[idx] for idx in outlier_indices]

    # if outlier_info:
    #     print("\nResumen de Outliers Detectados:")
    #     for idx, ratios in sorted(outlier_info.items()):
    #         print(f"  - Segmento {idx}: Outlier en {', '.join(ratios)}")

    # Filtramos los segmentos si deleteOutliers=True
    if deleteOutliers:
        filtered_segments = [segment for idx, segment in enumerate(segments_EEG) if idx not in outlier_indices]
        print(f"\nSe eliminaron {len(outlier_segments)} outliers. Segmentos restantes: {len(filtered_segments)}")
    else:
        filtered_segments = segments_EEG

    # Guardar gráficos si está activado
    if saveFigures:
        save_outlier_figures(outlier_segments, outlier_info, bands_list, pathFigures)

    return filtered_segments, outlier_segments, outlier_info

def detect_and_plot_outliers_2(segments_EEG, ratio_names, bands_list, threshold_percentile=2, saveFigures=False, pathFigures="", deleteOutliers=True):
    
    outlier_info = {}
    outlier_indices = set()

    # SEPARADO CADA LABEL
    segments_by_label = {0: [], 1: []}
    for segment in segments_EEG:
        label = segment['Label']
        segments_by_label[label].append(segment)

    for ratio_name in ratio_names:
        for label in [0, 1]:
            ratio_values = [segment[ratio_name] for segment in segments_by_label[label] if ratio_name in segment]
            
            if not ratio_values:
                print(f"No se encontraron valores para {ratio_name} en segmentos con Label={label}.")
                continue  

            lower_threshold = np.percentile(ratio_values, threshold_percentile)
            upper_threshold = np.percentile(ratio_values, 100 - threshold_percentile)

            print(f"Thresholds para {ratio_name} (Label={label}): Inferior={lower_threshold:.4f}, Superior={upper_threshold:.4f}")

            for idx, segment in enumerate(segments_EEG):
                if segment['Label'] == label and ratio_name in segment:
                    if segment[ratio_name] < lower_threshold or segment[ratio_name] > upper_threshold:
                        outlier_indices.add(idx)
                        if idx not in outlier_info:
                            outlier_info[idx] = []
                        outlier_info[idx].append(ratio_name)

    outlier_segments = [segments_EEG[idx] for idx in outlier_indices]

    # if outlier_info:
    #     print("\nResumen de Outliers Detectados:")
    #     for idx, ratios in sorted(outlier_info.items()):
    #         print(f"  - Segmento {idx}: Outlier en {', '.join(ratios)}")

    if deleteOutliers:
        filtered_segments = [segment for idx, segment in enumerate(segments_EEG) if idx not in outlier_indices]
        print(f"\nSe eliminaron {len(outlier_segments)} outliers. Segmentos restantes: {len(filtered_segments)}")
    else:
        filtered_segments = segments_EEG

    # Guardar gráficos si está activado
    if saveFigures:
        save_outlier_figures(outlier_segments, outlier_info, bands_list, pathFigures)

    return filtered_segments, outlier_segments, outlier_info

def save_outlier_figures(outlier_segments, outlier_info, bands_list, path):
    band_colors = {
        "delta": "red",
        "theta": "blue",
        "alpha": "green",
        "sigma": "purple",
        "beta": "orange"
    }

    for idx, segment in enumerate(outlier_segments):
        segment_idx = list(outlier_info.keys())[idx]  # Obtener el índice original
        signal = segment['Signal']
        time = np.linspace(0, len(signal) / segment['SamplingRate'], len(signal))

        ratios_detected = outlier_info[segment_idx]

        for ratio_name in ratios_detected:
            output_folder = os.path.join(path + "outliers", ratio_name)
            os.makedirs(output_folder, exist_ok=True)

            bands = ratio_name.replace("Ratio_", "").split("_")
            energy_p = segment.get(f'Energy_{bands[0]}', 'N/A')
            energy_q = segment.get(f'Energy_{bands[1]}', 'N/A')
            ratio_value = segment.get(ratio_name, 'N/A')

            freqs = np.fft.rfftfreq(len(signal), 1/segment['SamplingRate'])
            fft_magnitude = np.abs(np.fft.rfft(signal))

            band_p_signal = bandpass_filter(signal, segment['SamplingRate'], bands_list[bands[0]][0], bands_list[bands[0]][1])
            band_q_signal = bandpass_filter(signal, segment['SamplingRate'], bands_list[bands[1]][0], bands_list[bands[1]][1])

            fig = plt.figure(figsize=(8, 7))
            gs = gridspec.GridSpec(4, 1, height_ratios=[6, 6, 6, 1])

            label_map = {
                0: "sin apnea",
                1: "con apnea"
            }

            # Señal en el dominio del tiempo
            ax0 = plt.subplot(gs[0])
            ax0.plot(time, signal, label="EEG Signal")
            ax0.axhline(y=0, color='r', linestyle='--', label="Baseline")
            ax0.set_title(f"Segmento {segment_idx}: {label_map[segment['Label']]}")
            ax0.set_xlabel("Tiempo (s)")
            ax0.set_ylabel("Amplitud")
            ax0.legend()

            # Espectro de frecuencia
            ax1 = plt.subplot(gs[1])
            ax1.plot(freqs, fft_magnitude, color='black', label="FFT Magnitude")
            for band, (low, high) in bands_list.items():
                ax1.axvspan(low, high, color=band_colors.get(band, "gray"), alpha=0.3, label=f"{band} band")
            ax1.set_title("FFT")
            ax1.set_xlabel("Frecuencia (Hz)")
            ax1.set_ylabel("Magnitud")
            ax1.set_xlim(0, bands_list["beta"][1])
            ax1.legend()

            # Señales filtradas en bandas
            ax2 = plt.subplot(gs[2])
            ax2.plot(time, band_p_signal, label=f"{bands[0]} band", color=band_colors.get(bands[0], "gray"))
            ax2.plot(time, band_q_signal, label=f"{bands[1]} band", color=band_colors.get(bands[1], "gray"))
            ax2.set_title("Segmento filtrado en bandas")
            ax2.set_xlabel("Tiempo (s)")
            ax2.set_ylabel("Amplitud")
            ax2.legend()

            # Información de outlier
            ax3 = plt.subplot(gs[3])
            ax3.axis('off')
            text_info = (f"Outlier en {ratio_name}\n"
                         f"Ratio Value: {ratio_value:.4f}\n"
                         f"Energía {bands[0]}: {energy_p:.4f}\n"
                         f"Energía {bands[1]}: {energy_q:.4f}")
            ax3.text(0.5, 0.5, text_info, fontsize=10, verticalalignment='top', horizontalalignment='center',
                     bbox=dict(facecolor='white'))

            plt.tight_layout()
            filename = os.path.join(output_folder, f"outlier_{segment_idx}.png")
            plt.savefig(filename, dpi=300)
            plt.close(fig)

def plot_energy_ratios_boxplot(segments_EEG, selected_ratios = []):

    if selected_ratios == []:
        selected_ratios += [key for key in segments_EEG[0] if key.startswith('Ratio')]
    df = pd.DataFrame(segments_EEG)
    df_melted = df.melt(id_vars=['Label'], value_vars=selected_ratios, var_name='Energy Ratio', value_name='Value')

    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Energy Ratio', y='Value', hue='Label', data=df_melted, palette='Set2',
                flierprops={'marker': '_', 'color': 'red', 'markersize': 15, 'linestyle': 'none'})

    plt.xlabel('')
    plt.ylabel('Energy Ratio Values')
    plt.title('Energy Ratios for Different Labels')
    plt.legend(title='Label', loc='upper right')

    plt.xticks(rotation=15)
    plt.grid(which='both', axis='y', linestyle='--', alpha=0.7)
    plt.show()

def plot_energy_ratio_variation(segments_EEG, energy_ratio = 'Ratio_delta_beta'):

    df = pd.DataFrame(segments_EEG)
    segment_numbers = range(len(df))
    df_label_0 = df[df['Label'] == 0]
    df_label_1 = df[df['Label'] == 1]

    plt.figure(figsize=(12, 6))
    plt.scatter(df_label_0.index, df_label_0[energy_ratio], marker='*', edgecolors='blue', facecolors='none', label='Label 0')
    plt.scatter(df_label_1.index, df_label_1[energy_ratio], marker='o', edgecolors='red', facecolors='none', label='Label 1')

    plt.xlabel('Segment Number')
    plt.ylabel(f'{energy_ratio} Value')
    plt.title(f'Variation of {energy_ratio} by Segment')
    plt.legend()

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(13, 6))
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['without apnea', 'with apnea'])
    cm_display.plot(cmap='Blues', ax=ax)
    for text in ax.texts:
        text.set_visible(False)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = ax.text(j, i, f'{cm[i, j]}', ha='center', va='center', color='black')
    ax.set_title("Confusion Matrix")
    plt.show()

def open_saved_figure(filename):
    with open(filename, "rb") as f:
        fig = pickle.load(f)
    fig.show()
    plt.show()

open_saved_figure("C:\dev\ApneaDetection2\src KNN\segments_filtered\segment_1.pkl")
#main_single_file()
#main_multiple_files()