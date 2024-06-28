from LecturaSenalesReales import *
from LecturaAnotaciones import Anotaciones
# columns=['Cantidad de apneas', 'Longitud promedio apneas', 'Cantidad de segmentos de 30 seg sin overlap', 'Con apnea', 'Sin apnea', 'Cantidad de segmentos de 30 seg con overlap de 10', 'Con apnea', 'Sin apnea']
columns=['Cantidad de apneas', 'Longitud promedio apneas', 'Cantidad de segmentos', 'Con apnea', 'Sin apnea']

statistics = pd.DataFrame(columns = columns)

for i in range(1, 284): 
    try:
        EDF_path = f"C:\\Users\\elena\\OneDrive\\Documentos\\Tesis\\Dataset\\HomePAP\\polysomnography\\edfs\\lab\\full\\homepap-lab-full-1600{i:03d}.edf"
        all_signals = read_signals_EDF(EDF_path)
        annotations_path = f"C:\\Users\\elena\\OneDrive\\Documentos\\Tesis\\Dataset\\HomePAP\\polysomnography\\annotations-events-profusion\\lab\\full\\homepap-lab-full-1600{i:03d}-profusion.xml"
        annotations = Anotaciones(annotations_path)
    except:
        continue
    
    len_prom = 0
    cant_apneas = 0
    for event in annotations:
        for annotation in annotations[event]:
            len_prom += annotation[1]
            cant_apneas += 1
    if(cant_apneas > 0):
        len_prom = len_prom / cant_apneas
    try:
        #segmentos3010 = get_signal_segments(all_signals['M1']['Signal'], all_signals['M1']['Time'], all_signals['M1']['SamplingRate'], annotations, 30, 10)
        #segmentos30 = get_signal_segments(all_signals['M1']['Signal'], all_signals['M1']['Time'], all_signals['M1']['SamplingRate'], annotations, 30, 0)
        segmentosstrict = get_signal_segments_strict(all_signals['C3']['Signal'], all_signals['C3']['Time'], all_signals['C3']['SamplingRate'], annotations)
    except:
        continue
    # con_apnea_3010 =  0
    # for segmento in segmentos3010:
    #     con_apnea_3010 += segmento['Label']
    # sin_apnea_3010 = len(segmentos3010) - con_apnea_3010

    # con_apnea_30 =  0
    # for segmento in segmentos30:
    #     con_apnea_30 += segmento['Label']
    # sin_apnea_30 = len(segmentos30) - con_apnea_30

    con_apnea_strict =  0
    for segmento in segmentosstrict:
        con_apnea_strict += segmento['Label']
    sin_apnea_strict = len(segmentosstrict) - con_apnea_strict

    # statistic = pd.DataFrame([[cant_apneas, len_prom, len(segmentos30), con_apnea_30, sin_apnea_30, len(segmentos3010), con_apnea_3010, sin_apnea_3010]], columns=columns, index=[f"homepap-lab-full-1600{i:03d}"])
    statistic = pd.DataFrame([[cant_apneas, len_prom, len(segmentosstrict), con_apnea_strict, sin_apnea_strict]], columns=columns, index=[f"homepap-lab-full-1600{i:03d}"])
    print(statistic)
    statistics = pd.concat([statistics, statistic]) 

statistics.to_excel("statistics_strict.xlsx")
