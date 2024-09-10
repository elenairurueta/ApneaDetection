from Imports import *
from pyedflib import EdfReader

def read_signals_EDF(path:str):
    EDF = EdfReader(path)
    labels = EDF.getSignalLabels()
    all_signals = {} 
    for channel in range(0, EDF.signals_in_file):
        signal = EDF.readSignal(channel)
        eje_x = np.arange(0, EDF.getFileDuration(), 1/EDF.getSampleFrequency(channel))
        all_signals[labels[channel]] = {'Signal': signal, 'Time': eje_x, 'Dimension': EDF.getPhysicalDimension(channel), 'SamplingRate': EDF.getSampleFrequency(channel)}
    
    return all_signals

def get_bipolar_signal(signal1, signal2):
    if(signal1['SamplingRate'] != signal2['SamplingRate']):
        raise TypeError("Signals not compatible: different sampling rate")
    signal = signal1['Signal'] - signal2['Signal']
    tiempo = signal1['Time']
    sampling = signal1['SamplingRate']
    return signal, tiempo, sampling
    
def get_signal_segments(signal, tiempo, sampling_rate, annotations, period_length=30, overlap=10, perc_apnea=0.3):
    step = period_length - overlap
    num_periods = int(np.ceil((tiempo[-1] - period_length) / step))+1
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
        segments.append({'Signal': signal[start_index:end_index], 'Label': label, 'Start': period_start, 'End': period_end, 'SamplingRate': sampling_rate})
    if(len(segments[-1]['Signal']) < len(segments[0]['Signal'])):
        segments.pop()
    return segments

def get_signal_segments_strict(signal, tiempo, sampling_rate, annotations):
    all_annotations = []
    segments_conapnea = []
    segments_sinapnea = []

    for event in annotations:
        all_annotations = all_annotations + annotations[event]
    all_annotations.sort(key=lambda x: x[0]) #ordeno por inicio de la anotación
    for idx, annotation in enumerate(all_annotations):
            inicio_apnea, duracion_apnea = annotation
            fin_apnea = inicio_apnea + duracion_apnea
            if(idx < len(all_annotations) - 1):
                inicio_prox = all_annotations[idx + 1][0]
                if (fin_apnea + 15) <= inicio_prox:
                    #tomo 15 seg antes y 15 seg después
                    start_time = fin_apnea - 15
                    start_index = int(start_time * sampling_rate)
                    end_time = fin_apnea + 15
                    end_index = int(end_time * sampling_rate)
                    segments_conapnea.append({'Signal': signal[start_index:end_index], 'Label': 1, 'Start': start_time, 'End': end_time, 'SamplingRate': sampling_rate})
                    if(len(segments_conapnea[-1]['Signal']) < len(segments_conapnea[0]['Signal'])):
                        segments_conapnea.pop()

    t = tiempo[0]
    while(t<tiempo[-1]):
        tieneapnea = False
        for annotation in all_annotations:
            inicio_apnea, duracion_apnea = annotation
            fin_apnea = inicio_apnea + duracion_apnea
            if(((inicio_apnea >= t) and (inicio_apnea <= t + 45)) or
               ((fin_apnea >= t) and (fin_apnea <= t + 45)) or
               ((inicio_apnea < t) and (fin_apnea > t + 45))):
                tieneapnea = True
                t = fin_apnea + 1
                break
        if(tieneapnea == False):
            start_index = int((t+15) * sampling_rate)
            end_index = int((t+45) * sampling_rate)
            segments_sinapnea.append({'Signal': signal[start_index:end_index], 'Label': 0, 'Start': t+15, 'End': t+45, 'SamplingRate': sampling_rate})
            if(len(segments_sinapnea[-1]['Signal']) < len(segments_sinapnea[0]['Signal'])):
                segments_sinapnea.pop()
            t = t + 30
    

    segments = segments_conapnea + segments_sinapnea
    segments.sort(key=lambda x: x['Start'])
    random.shuffle(segments)

    return segments