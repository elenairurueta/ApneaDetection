from scipy.signal import filtfilt, firwin, firwin2, butter, freqz
import matplotlib.pyplot as plt
import numpy as np

def filter_FIR(x, fs):

    nyquist = fs / 2
    lowcut = 0.5
    highcut = 80
    orden = 101
    coef_fir = firwin(orden, [lowcut/nyquist, highcut/nyquist], pass_zero=False) #the frequencies in cutoff should be positive and monotonically increasing between 0 and fs/2
    
    w, h = freqz(coef_fir)

    # plt.figure(1, figsize=(10, 6))
    # plt.subplot(2, 1, 1)
    # plt.plot(0.5 * fs * w / np.pi, 20 * np.log10(abs(h)), 'b')
    # plt.title('Respuesta en frecuencia del filtro FIR (Magnitud)')
    # plt.xlabel('Frecuencia [Hz]')
    # plt.ylabel('Ganancia [dB]')
    # plt.grid()

    # plt.subplot(2, 1, 2)
    # plt.plot(0.5 * fs * w / np.pi, np.angle(h), 'g')
    # plt.title('Respuesta en frecuencia del filtro FIR (Ángulo de fase)')
    # plt.xlabel('Frecuencia [Hz]')
    # plt.ylabel('Ángulo [radianes]')
    # plt.grid()

    # plt.tight_layout()
    # plt.show()

    return filtfilt(coef_fir, 1.0, x)

def filter_FIR2(x, fs, plot = True):

    delta_s = -40

    def db2mag(db):
        """Convierte de decibelios a magnitud lineal."""
        return 10**(db / 20)

    #orden = 2001
    orden = 5001
    #freq_points = np.array([0, 0.3, 1, 80, 90, fs/2])
    #freq_points = np.array([0, 0.01, 0.1, 80, 90, fs/2]) 
    #freq_points = np.array([0, 0.01, 0.05, 80, 90, fs/2])
    #freq_points = np.array([0, 0.01, 0.04, 80, 90, fs/2])
    freq_points = np.array([0, 0.005, 0.01, 80, 90, fs/2])

    gain_points = np.array([db2mag(delta_s), db2mag(delta_s), 1, 1, db2mag(delta_s), db2mag(delta_s)])

    coef_fir = firwin2(orden, freq_points, gain_points, fs=fs)
    
    np.savetxt('coef_fir5.txt', coef_fir, delimiter='\n')
    
    if(plot):
        w, h = freqz(coef_fir)

        plt.figure(1, figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.plot(0.5 * fs * w / np.pi, 20 * np.log10(abs(h)), 'b', marker='o', markersize=3)
        plt.title('Respuesta en frecuencia del filtro FIR2 (Magnitud)')
        plt.xlabel('Frecuencia [Hz]')
        plt.xlim([0, 100])
        plt.ylabel('Ganancia [dB]')
        plt.grid(which='both', linestyle='--', linewidth=0.5)

        plt.grid()

        plt.subplot(2, 1, 2)
        plt.plot(0.5 * fs * w / np.pi, np.angle(h), 'g')
        plt.title('Respuesta en frecuencia del filtro FIR2 (Ángulo de fase)')
        plt.xlabel('Frecuencia [Hz]')
        plt.xlim([0, 100])
        plt.ylabel('Ángulo [radianes]')
        plt.grid()

        plt.tight_layout()
        plt.show()

    return filtfilt(coef_fir, 1.0, x)


def filter_Butter(x, fs, plot = True):

    nyquist = fs / 2
    lowcut = 0.01 # 0.5 
    highcut = 80
    orden = 4
    b, a = butter(orden, [lowcut / nyquist, highcut / nyquist], btype='band')

    if(plot):
        w, h = freqz(b, a, worN=8000)
        
        plt.figure(1, figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.plot(0.5 * fs * w / np.pi, 20 * np.log10(abs(h)), 'b')
        plt.title('Respuesta en frecuencia del filtro Butterworth (Magnitud) [dB]')
        plt.xlabel('Frecuencia [Hz]')
        plt.ylabel('Ganancia')
        plt.grid()

        plt.subplot(2, 1, 2)
        plt.plot(0.5 * fs * w / np.pi, np.angle(h), 'g')
        plt.title('Respuesta en frecuencia del filtro Butterworth (Ángulo de fase)')
        plt.xlabel('Frecuencia [Hz]')
        plt.ylabel('Ángulo [radianes]')
        plt.grid()

        plt.tight_layout()
        plt.show()
    
    return filtfilt(b, a, x) #padtype='constant', method='pad'

def filter_Notch_FIR(x, fs, plot = True):
    
    f1 = 50.0
    f2 = 60.0

    nyquist = fs / 2

    orden = 601

    bw = 0.7

    notch_fir_50 = firwin(orden, [(f1 - bw) / nyquist, (f1 + bw) / nyquist], pass_zero=True)
    notch_fir_60 = firwin(orden, [(f2 - bw) / nyquist, (f2 + bw) / nyquist], pass_zero=True)

    x_filtered_50 = filtfilt(notch_fir_50, 1.0, x)
    x_filtered_60 = filtfilt(notch_fir_60, 1.0, x_filtered_50)

    if(plot):
        w_50, h_50 = freqz(notch_fir_50)
        w_60, h_60 = freqz(notch_fir_60)

        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.plot(0.5 * fs * w_50 / np.pi, 20 * np.log10(abs(h_50)), 'b')
        plt.title('Respuesta en frecuencia del filtro Notch FIR 50 Hz')
        plt.xlabel('Frecuencia [Hz]')
        plt.ylabel('Ganancia [dB]')
        plt.grid()

        plt.subplot(2, 1, 2)
        plt.plot(0.5 * fs * w_60 / np.pi, 20 * np.log10(abs(h_60)), 'g')
        plt.title('Respuesta en frecuencia del filtro Notch FIR 60 Hz')
        plt.xlabel('Frecuencia [Hz]')
        plt.ylabel('Ganancia [dB]')
        plt.grid()

        plt.tight_layout()
        plt.show()

    return x_filtered_60


def plot_spectrum(x, sr, title):
    X = np.fft.fft(x)
    freqs = np.fft.fftfreq(len(x), 1/sr)
    
    plt.plot(freqs[:len(freqs)//2], np.abs(X[:len(X)//2]))  # Solo la mitad positiva
    plt.title(title)
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("Magnitud")
    plt.grid()
