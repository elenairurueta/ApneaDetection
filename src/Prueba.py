from Imports import *
from RealSignals import *
from Annotations import get_annotations
from Filtering import filter_FIR, filter_FIR2, filter_Butter, filter_Notch_FIR, plot_spectrum
from PlotSignals import plot_signals

path_annot = "C:/Users/elena/OneDrive/Documentos/Tesis/Dataset/HomePAP/polysomnography/annotations-events-profusion/lab/full"
path_edf = "C:/Users/elena/OneDrive/Documentos/Tesis/Dataset/HomePAP/polysomnography/edfs/lab/full"

files = [4, 43, 53, 55, 63, 72, 84, 95, 105, 113, 122, 151]
for file in files:
        
    all_signals = read_signals_EDF(path_edf + f"\\homepap-lab-full-1600{str(file).zfill(3)}.edf")
    annotations = get_annotations(path_annot + f"\\homepap-lab-full-1600{str(file).zfill(3)}-profusion.xml")
    bipolar_signal, time, sampling = get_bipolar_signal(all_signals['C3'], all_signals['O1'])

    #filtered_signal = filter_Butter(bipolar_signal, sampling) #, plot = False)
    filtered_signal_1 = filter_FIR2(bipolar_signal, sampling) #, plot = False)
    filtered_signal = filter_Notch_FIR(filtered_signal_1, sampling) #, plot = False)
    
    plt.figure(2, figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plot_spectrum(bipolar_signal, sampling, f"Espectro de la señal original - 1600{str(file).zfill(3)}")

    plt.subplot(1, 2, 2)
    plot_spectrum(filtered_signal, sampling, f"Espectro de la señal filtrada con FIR + Notch - 1600{str(file).zfill(3)}")

    plt.tight_layout()
    plt.show()
    # PATH = f'D:\models\FILTERING\espectros FIR Notch\\espectros FIR Notch 00{str(file).zfill(3)}_001_80.png'
    # plt.savefig(PATH)
    # plt.close()

    plot_signals(annotations, time, bipolar_signal, f'Original - 1600{str(file).zfill(3)}', filtered_signal_1, f'FIR - 1600{str(file).zfill(3)}', filtered_signal, f'FIR + Notch - 1600{str(file).zfill(3)}')
