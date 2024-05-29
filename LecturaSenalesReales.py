from Imports import *
from pyedflib import EdfReader
from LecturaAnotaciones import *

def read_signals_EDF(EDF):
    labels = EDF.getSignalLabels()
    all_signals = {} 
    for channel in range(5,9):#(0, EDF.signals_in_file):
        signal = EDF.readSignal(channel)
        eje_x = np.arange(0, EDF.getFileDuration(), 1/EDF.getSampleFrequency(channel))
        all_signals[labels[channel]] = {'Senal': signal, 'Tiempo': eje_x, 'Unidad': EDF.getPhysicalDimension(channel), 'SamplingRate': EDF.getSampleFrequency(channel)}
    return all_signals

def plot_signal(key):
    signal = all_signals.get(key)
    plt.plot(signal['Tiempo'], signal['Senal'])
    plt.title(key)
    plt.xlabel('t')
    plt.ylabel(signal['Unidad'])
    plt.xlim(0, signal['Tiempo'][-1])
    plt.show() 

def plot_bipolar_signal(key1, key2, t = 'min', annotations = None):
    """
    t puede ser min, h, seg
    Las señales deben tener el mismo vector de tiempo y la misma unidad.
    """
    signal1 = all_signals.get(key1)
    signal2 = all_signals.get(key2)
    signal = signal1['Senal'] - signal2['Senal']
    if((signal1['Tiempo'] == signal2['Tiempo']).all() and (signal1['Unidad'] == signal2['Unidad'])):
        tiempo = signal1['Tiempo']
        if(t == 'min'):
            tiempo = tiempo/60
        elif(t == 'h'):
            tiempo = tiempo/3600

        # plt.plot(tiempo, signal)
        # plt.title(key1 + '-' + key2)
        # plt.xlabel('t['+t+']')
        # plt.xticks(tiempo)
        # plt.ylabel(signal1['Unidad'])
        # plt.xlim(0, tiempo[-1])
        # plt.show() 

        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.25)
        l, = plt.plot(tiempo, signal)
        plt.title(key1 + '-' + key2)
        plt.xlabel(f't[{t}]')
        plt.ylabel(signal1['Unidad'])
        plt.xlim(0, tiempo[-1])
        plt.ylim(min(signal1['Senal']), max(signal1['Senal']))


        axcolor = 'lightgoldenrodyellow'
        axslider = plt.axes([0.124, 0.1, 0.776, 0.03], facecolor=axcolor)
        step = {'h': 0.1, 'min': 1, 'seg': 10}
        slider = Slider(
            ax = axslider,  
            label = '',
            valmin = 0, 
            valmax = tiempo[-1]-1, 
            valinit=0, 
            valstep=step[t])

        def update(val):
            inicio = slider.val
            ax.set_xlim(inicio, inicio + 1)
            fig.canvas.draw_idle()
    
        slider.on_changed(update)

        resetax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
        button = Button(resetax, 'Reset', hovercolor='0.975')

        def reset(event):
            slider.reset()
        button.on_clicked(reset)

        if(annotations != None):
            for annotation in annotations:
                start, duration = annotation
                if(t == 'min'):
                    start = start/60
                    duration = duration/60
                elif(t == 'h'):
                    start = start/3600
                    duration = duration/3600
                ax.axvspan(start, start + duration, color='red', alpha=0.3, label='Obstructive Apnea')

        plt.show()


    else:
        print('Senales no compatibles')
    
def get_bipolar_signal(key1, key2):
    """
    t puede ser min, h, seg
    Las señales deben tener el mismo vector de tiempo y la misma unidad.
    """
    signal1 = all_signals.get(key1)
    signal2 = all_signals.get(key2)
    signal = signal1['Senal'] - signal2['Senal']
    tiempo = signal1['Tiempo']
    sampling = signal1['SamplingRate']
    return signal, tiempo, sampling
    
def get_signal_segments(signal, tiempo, sampling_rate, annotations, period_length=30, overlap=10):
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
            start, duration = annotation
            apnea_start = start
            apnea_end = start + duration
            
            if (apnea_start < period_end and apnea_end > period_start):
                label = 1
                break
        segments.append({'Senal': signal[start_index:end_index], 'Label': label, 'Start': period_start, 'End': period_end, 'SamplingRate': sampling_rate})
    return segments


def plot_apnea_segments(segments):
    plt.figure(1)
    for segment in segments:
        if(segment['Label'] == 1):
            tiempo_segmento = np.arange(0, 30, 1/segment['SamplingRate'])
            plt.plot(tiempo_segmento, segment['Senal'])
            start = segment['Start']
            end = segment['End']
            label = segment['Label']
            plt.title(f'Senal: {start} a {end}seg - Label: {label}')
            plt.show()

def plot_all_segments(segments):
    plt.figure(1)
    for segment in segments:
        tiempo_segmento = np.arange(0, 30, 1/segment['SamplingRate'])
        plt.plot(tiempo_segmento, segment['Senal'])
        start = segment['Start']
        end = segment['End']
        label = segment['Label']
        plt.title(f'Senal: {start} a {end}seg - Label: {label}')
        plt.show()


all_signals = read_signals_EDF(EdfReader('homepap-lab-full-1600001.edf'))
print(all_signals.keys())

annotations = Anotaciones()
#plot_bipolar_signal('C4', 'M1', 'min', annotations)
senalbipolar, tiempo, sampling = get_bipolar_signal('C4', 'M1')
segments = get_signal_segments(senalbipolar, tiempo, sampling, annotations)
plot_apnea_segments(segments)
#plot_all_segments(segments)