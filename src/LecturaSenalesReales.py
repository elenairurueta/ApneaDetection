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

def plot_signal(key):
    signal = all_signals.get(key)
    plt.plot(signal['Time'], signal['Signal'])
    plt.title(key)
    plt.xlabel('t')
    plt.ylabel(signal['Dimension'])
    plt.xlim(0, signal['Time'][-1])
    plt.show() 

def plot_bipolar_signal(signal1, signal2, t = 'min', annotations = None):
    """
    t puede ser min, h, seg
    Las señales deben tener el mismo vector de tiempo y la misma unidad.
    """
    signal = signal1['Signal'] - signal2['Signal']
    if((signal1['Time'] == signal2['Time']).all() and (signal1['Dimension'] == signal2['Dimension'])):
        tiempo = signal1['Time']
        if(t == 'min'):
            tiempo = tiempo/60
        elif(t == 'h'):
            tiempo = tiempo/3600

        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.25)
        plt.subplot(3,1,1)
        plt.plot(tiempo, signal1['Signal'])
        plt.subplot(3,1,2)
        plt.plot(tiempo, signal2['Signal'])
        plt.subplot(3,1,3)
        plt.plot(tiempo, signal)

        plt.title('bipolar')
        plt.xlabel(f't[{t}]')
        plt.ylabel(signal1['Dimension'])
        plt.xlim(0, tiempo[-1])
        plt.ylim(min(signal1['Signal']), max(signal1['Signal']))

        axcolor = 'lightgoldenrodyellow'
        axslider = plt.axes([0.124, 0.1, 0.776, 0.03], facecolor=axcolor)
        step = {'h': 0.01, 'min': 0.1, 'seg': 1}
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

        colors = {
            'Obstructive Apnea': 'red',
            'Central Apnea': 'blue',
            'Mixed Apnea': 'green',
            'Hypopnea': 'orange',
            'Obstructive Hypopnea': 'purple',
            'Central Hypopnea': 'brown',
            'Mixed Hypopnea': 'pink'
        }
        if annotations is not None:
            for evento, evento_annotations in annotations.items():
                color = colors.get(evento) if colors else 'red'
                for annotation in evento_annotations:
                    start, duration = annotation
                    if t == 'min':
                        start = start / 60
                        duration = duration / 60
                    elif t == 'h':
                        start = start / 3600
                        duration = duration / 3600
                    ax.axvspan(start, start + duration, color=color, alpha=0.3, label=evento)
        
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

        plt.show()

    else:
        print('Not compatible signals')
    
def get_bipolar_signal(signal1, signal2):
    signal = signal1['Signal'] - signal2['Signal']
    tiempo = signal1['Time']
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
            for anot in annotations[annotation]:
                start, duration = anot
                apnea_start = start
                apnea_end = start + duration
                
                if (apnea_start < period_end and apnea_end > period_start):
                    order = [period_start, apnea_start, apnea_end, period_end]
                    order.sort()
                    apnea_in_segment = order[2] - order[1]
                    if(apnea_in_segment >= 0.3*duration):
                        label = 1
                    break
        segments.append({'Signal': signal[start_index:end_index], 'Label': label, 'Start': period_start, 'End': period_end, 'SamplingRate': sampling_rate})
    return segments

def plot_apnea_segments(segments):
    plt.figure(1)
    for segment in segments:
        if(segment['Label'] == 1):
            tiempo_segmento = np.arange(0, 30, 1/segment['SamplingRate'])
            plt.plot(tiempo_segmento, segment['Signal'])
            start = segment['Start']
            end = segment['End']
            label = segment['Label']
            plt.title(f'Signal: {start} to {end}seg - Label: {label}')
            plt.show()

def plot_all_segments(segments):
    plt.figure(1)
    for segment in segments:
        tiempo_segmento = np.arange(0, 30, 1/segment['SamplingRate'])
        plt.plot(tiempo_segmento, segment['Signal'])
        start = segment['Start']
        end = segment['End']
        label = segment['Label']
        plt.title(f'Senal: {start} a {end}seg - Label: {label}')
        plt.show()
