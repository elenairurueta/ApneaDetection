from RealSignals import *
from Annotations import get_annotations
import openpyxl

def statistics_apneas():

    # columns=['Cantidad de apneas', 'Longitud promedio apneas', 'Cantidad de segmentos de 30 seg sin overlap', 'Con apnea', 'Sin apnea', 'Cantidad de segmentos de 30 seg con overlap de 10', 'Con apnea', 'Sin apnea']
    columns=['Cantidad de apneas', 'Longitud promedio apneas', 'Segmentos de 30 seg con overlap de 10seg con al menos 10% de apnea', 'Con apnea', 'Sin apnea', 'Segmentos de 30 seg sin overlap con al menos 10% de apnea', 'Segmentos de 30 seg con overlap de 10seg con al menos 30% de apnea', 'Con apnea', 'Sin apnea', 'Segmentos de 30 seg sin overlap con al menos 30% de apnea', 'Con apnea', 'Sin apnea', 'Segmentos de 30 seg con overlap de 10seg con al menos 50% de apnea', 'Con apnea', 'Sin apnea', 'Segmentos de 30 seg sin overlap con al menos 50% de apnea', 'Con apnea', 'Sin apnea', 'Segmentos de 30 seg con overlap de 10seg con al menos 70% de apnea', 'Con apnea', 'Sin apnea', 'Segmentos de 30 seg sin overlap con al menos 70% de apnea', 'Con apnea', 'Sin apnea', 'Segmentos de 30 seg con overlap de 10seg con al menos 90% de apnea', 'Con apnea', 'Sin apnea', 'Segmentos de 30 seg sin overlap con al menos 90% de apnea', 'Con apnea', 'Sin apnea', 'Con apnea', 'Sin apnea', 'Cantidad de segmentos strict', 'Con apnea', 'Sin apnea']

    statistics = pd.DataFrame(columns = columns)

    for i in range(1, 374): 
        try:
            EDF_path = f"C:\\Users\\elena\\OneDrive\\Documentos\\Tesis\\Dataset\\HomePAP\\polysomnography\\edfs\\lab\\full\\homepap-lab-full-1600{i:03d}.edf"
            all_signals = read_signals_EDF(EDF_path)
            annotations_path = f"C:\\Users\\elena\\OneDrive\\Documentos\\Tesis\\Dataset\\HomePAP\\polysomnography\\annotations-events-profusion\\lab\\full\\homepap-lab-full-1600{i:03d}-profusion.xml"
            annotations = get_annotations(annotations_path)
        except:
            #print(f'\nhomepap-lab-full-1600{i:03d} no se pudo leer')
            continue
        
        len_prom = 0
        cant_apneas = 0
        for event in annotations:
            for annotation in annotations[event]:
                len_prom += annotation[1]
                cant_apneas += 1
        if(cant_apneas > 0):
            try:
                len_prom = len_prom / cant_apneas
                segmentos_30_10_10 = get_signal_segments(all_signals['M1']['Signal'], all_signals['M1']['Time'], all_signals['M1']['SamplingRate'], annotations, 30, 10, 0.1)
                segmentos_30_0_10 = get_signal_segments(all_signals['M1']['Signal'], all_signals['M1']['Time'], all_signals['M1']['SamplingRate'], annotations, 30, 0, 0.1)
                segmentos_30_10_30 = get_signal_segments(all_signals['M1']['Signal'], all_signals['M1']['Time'], all_signals['M1']['SamplingRate'], annotations, 30, 10, 0.3)
                segmentos_30_0_30 = get_signal_segments(all_signals['M1']['Signal'], all_signals['M1']['Time'], all_signals['M1']['SamplingRate'], annotations, 30, 0, 0.3)
                segmentos_30_10_50 = get_signal_segments(all_signals['M1']['Signal'], all_signals['M1']['Time'], all_signals['M1']['SamplingRate'], annotations, 30, 10, 0.5)
                segmentos_30_0_50 = get_signal_segments(all_signals['M1']['Signal'], all_signals['M1']['Time'], all_signals['M1']['SamplingRate'], annotations, 30, 0, 0.5)
                segmentos_30_10_70 = get_signal_segments(all_signals['M1']['Signal'], all_signals['M1']['Time'], all_signals['M1']['SamplingRate'], annotations, 30, 10, 0.7)
                segmentos_30_0_70 = get_signal_segments(all_signals['M1']['Signal'], all_signals['M1']['Time'], all_signals['M1']['SamplingRate'], annotations, 30, 0, 0.7)
                segmentos_30_10_90 = get_signal_segments(all_signals['M1']['Signal'], all_signals['M1']['Time'], all_signals['M1']['SamplingRate'], annotations, 30, 10, 0.9)
                segmentos_30_0_90 = get_signal_segments(all_signals['M1']['Signal'], all_signals['M1']['Time'], all_signals['M1']['SamplingRate'], annotations, 30, 0, 0.9)
                segmentosstrict = get_signal_segments_strict(all_signals['C3']['Signal'], all_signals['C3']['Time'], all_signals['C3']['SamplingRate'], annotations)
            except:
                continue
        
        con_apnea_30_10_10 = sum(segmento['Label'] for segmento in segmentos_30_10_10)
        sin_apnea_30_10_10 = len(segmentos_30_10_10) - con_apnea_30_10_10

        con_apnea_30_0_10 = sum(segmento['Label'] for segmento in segmentos_30_0_10)
        sin_apnea_30_0_10 = len(segmentos_30_0_10) - con_apnea_30_0_10

        con_apnea_30_10_30 = sum(segmento['Label'] for segmento in segmentos_30_10_30)
        sin_apnea_30_10_30 = len(segmentos_30_10_30) - con_apnea_30_10_30

        con_apnea_30_0_30 = sum(segmento['Label'] for segmento in segmentos_30_0_30)
        sin_apnea_30_0_30 = len(segmentos_30_0_30) - con_apnea_30_0_30

        con_apnea_30_10_50 = sum(segmento['Label'] for segmento in segmentos_30_10_50)
        sin_apnea_30_10_50 = len(segmentos_30_10_50) - con_apnea_30_10_50

        con_apnea_30_0_50 = sum(segmento['Label'] for segmento in segmentos_30_0_50)
        sin_apnea_30_0_50 = len(segmentos_30_0_50) - con_apnea_30_0_50

        con_apnea_30_10_70 = sum(segmento['Label'] for segmento in segmentos_30_10_70)
        sin_apnea_30_10_70 = len(segmentos_30_10_70) - con_apnea_30_10_70

        con_apnea_30_0_70 = sum(segmento['Label'] for segmento in segmentos_30_0_70)
        sin_apnea_30_0_70 = len(segmentos_30_0_70) - con_apnea_30_0_70

        con_apnea_30_10_90 = sum(segmento['Label'] for segmento in segmentos_30_10_90)
        sin_apnea_30_10_90 = len(segmentos_30_10_90) - con_apnea_30_10_90

        con_apnea_30_0_90 = sum(segmento['Label'] for segmento in segmentos_30_0_90)
        sin_apnea_30_0_90 = len(segmentos_30_0_90) - con_apnea_30_0_90

        con_apnea_strict = sum(segmento['Label'] for segmento in segmentosstrict)
        sin_apnea_strict = len(segmentosstrict) - con_apnea_strict

        # con_apnea_3010 =  0
        # for segmento in segmentos3010:
        #     con_apnea_3010 += segmento['Label']
        # sin_apnea_3010 = len(segmentos3010) - con_apnea_3010

        # con_apnea_30 =  0
        # for segmento in segmentos30:
        #     con_apnea_30 += segmento['Label']
        # sin_apnea_30 = len(segmentos30) - con_apnea_30

        # con_apnea_strict =  0
        # for segmento in segmentosstrict:
        #     con_apnea_strict += segmento['Label']
        # sin_apnea_strict = len(segmentosstrict) - con_apnea_strict

        # statistic = pd.DataFrame([[cant_apneas, len_prom, len(segmentos30), con_apnea_30, sin_apnea_30, len(segmentos3010), con_apnea_3010, sin_apnea_3010]], columns=columns, index=[f"homepap-lab-full-1600{i:03d}"])
        statistic = pd.DataFrame([[cant_apneas, len_prom, 
                                   len(segmentos_30_10_10), con_apnea_30_10_10, sin_apnea_30_10_10, 
                                   len(segmentos_30_0_10), con_apnea_30_0_10, sin_apnea_30_0_10, 
                                   len(segmentos_30_10_30), con_apnea_30_10_30, sin_apnea_30_10_30, 
                                   len(segmentos_30_0_30), con_apnea_30_0_30, sin_apnea_30_0_30, 
                                   len(segmentos_30_10_50), con_apnea_30_10_50, sin_apnea_30_10_50, 
                                   len(segmentos_30_0_50), con_apnea_30_0_50, sin_apnea_30_0_50, 
                                   len(segmentos_30_10_70), con_apnea_30_10_70, sin_apnea_30_10_70, 
                                   len(segmentos_30_0_70), con_apnea_30_0_70, sin_apnea_30_0_70, 
                                   len(segmentos_30_10_90), con_apnea_30_10_90, sin_apnea_30_10_90, 
                                   len(segmentos_30_0_90), con_apnea_30_0_90, sin_apnea_30_0_90, 
                                   len(segmentosstrict), con_apnea_strict, sin_apnea_strict]], 
                                   columns=columns, index=[f"homepap-lab-full-1600{i:03d}"])
        print(statistic)
        statistics = pd.concat([statistics, statistic]) 

    statistics.to_excel("statistics_strict.xlsx")

def statistics_senales():
    columns=['E1',
    'E2', 
    'F3', 
    'F4',
    'C3', 
    'C4', 
    'O1', 
    'O2', 
    'M1', 
    'M2', 
    'Lchin', 
    'Rchin', 
    'Cchin', 
    'ECG1', 
    'ECG2', 
    'ECG3', 
    'Lleg', 
    'Rleg', 
    'Snore', 
    'CannulaFlow',
    'Airflow',
    'Chest', 
    'ABD', 
    'SUM', 
    'MaskFlow', 
    'Pressure', 
    'Pleth', 
    'Pulse', 
    'SAO2', 
    'Leak', 
    'Position', 
    'DHR']

    statistics = pd.DataFrame(columns = columns)
    for i in range(1, 284): 
        statistic = []
        try:
            EDF_path = f"C:\\Users\\elena\\OneDrive\\Documentos\\Tesis\\Dataset\\HomePAP\\polysomnography\\edfs\\lab\\full\\homepap-lab-full-1600{i:03d}.edf"
            all_signals = read_signals_EDF(EDF_path)
        except:
            continue
        for column in columns:
            try:
                signal = all_signals[column]
                statistic.append(all_signals[column]['SamplingRate'])
            except:
                statistic.append(0)
        reshaped_statistic = np.transpose(statistic).reshape(1, 32)
        statistic_df = pd.DataFrame(reshaped_statistic, columns=columns, index=[f"homepap-lab-full-1600{i:03d}"])
        print(statistic_df)
        statistics = pd.concat([statistics, statistic_df]) 

    statistics.to_excel("statistics_signals.xlsx")


statistics_apneas()