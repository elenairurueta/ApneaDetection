from Imports import *
from pyedflib import EdfReader

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
    
def get_signal_segments(signal, time, sampling_rate, annotations, period_length=30, overlap=10, perc_apnea=0.3):
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
    num_periods = int(np.ceil((time[-1] - period_length) / step))+1 #Calculates the number of periods based on the total signal duration, segment length, and overlap.
    segments = []

    for i in range(num_periods):
        period_start = i * step
        period_end = period_start + period_length   
        start_index = int(period_start * sampling_rate)
        end_index = int(period_end * sampling_rate)
        label = 0
        if end_index > len(signal):
            end_index = len(signal)

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
    if(len(segments[-1]['Signal']) < len(segments[0]['Signal'])):
        segments.pop()
    return segments

def get_signal_segments_strict(signal, tiempo, sampling_rate, annotations):
    """
    Extracts signal segments based on apnea annotations, creating two types of labeled segments:
    1. Segments with apnea (15 seconds before and 15 seconds after the apnea event)
    2. Segments without apnea (30-second windows where no apnea is present in the window or in the previous 15 seconds)

    Parameters:
    - signal: The full signal array from which to extract segments.
    - tiempo: Array of time points corresponding to the signal.
    - sampling_rate: The sampling rate of the signal (in Hz).
    - annotations: A dictionary where each key corresponds to an event type and the values are lists of apnea events (start time, duration).

    Returns:
    - A list of dictionaries, each containing:
    - 'Signal': The extracted signal segment.
    - 'Label': 1 for apnea segments, 0 for non-apnea segments.
    - 'Start': The start time of the segment.
    - 'End': The end time of the segment.
    - 'SamplingRate': The sampling rate used to extract the segment.
    """
    
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