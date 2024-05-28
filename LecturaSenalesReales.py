from Imports import *
from pyedflib import EdfReader


def plot_signal(key):
    signal = all_signals.get(key)
    plt.plot(signal['Tiempo'], signal['Senal'])
    plt.title(key)
    plt.xlabel('t')
    plt.ylabel(signal['Unidad'])
    plt.xlim(0, signal['Tiempo'][-1])
    plt.show() 


EDF = EdfReader('homepap-lab-full-1600001.edf')

labels = EDF.getSignalLabels()
all_signals = {} 
for channel in range(0, EDF.signals_in_file):
    signal = EDF.readSignal(channel)
    eje_x = np.arange(0, EDF.getFileDuration(), 1/EDF.getSampleFrequency(channel))
    all_signals[labels[channel]] = {'Senal': signal, 'Tiempo': eje_x, 'Unidad': EDF.getPhysicalDimension(channel)}

senales = all_signals.keys()
plot_signal('Chest')
