from Imports import *
import pyqtgraph as pg
from PyQt5.QtGui import QBrush, QColor
from RealSignals import read_signals_EDF, get_bipolar_signal
from Annotations import get_annotations



def plot_bipolar_signal(signal1, signal2, t = 'min', annotations = None):
    """
    t must be either 'min', 'h' or 'seg'
    Signals must have the same time vector and unit.
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

def plot_signals(annotations, tiempo, signal1, s1, signal2 = [ ], s2 = ' ', signal3 = [ ], s3 = ' '):

    colors = [
            [QColor(255, 255, 0, 100),  # Amarillo transparente
            QColor(255, 255, 0, 255)], # Amarillo 
            [QColor(128, 0, 128, 100),  # Violeta transparente
            QColor(128, 0, 128, 255)], # Violeta
            [QColor(255, 165, 0, 100),  # Naranja transparente
            QColor(255, 165, 0, 255)]  # Naranja
    ]

    win = pg.GraphicsLayoutWidget(show=True)
    win.setWindowTitle('Signals plot')

    win.showMaximized()
    win.ci.setBorder((50, 50, 100))

    pg.setConfigOptions(antialias=True)


    p1 = win.addPlot(title="Region Selection")
    p1.addLegend()
    p1.setDownsampling(auto=False, ds=4, mode='mean')
    p1.showGrid(x = True, y = True, alpha = 0.3)     
    p1.plot(tiempo, signal1 + 10, pen=(255,0,0), name = s1)
    if(len(signal2) > 0):
        p1.plot(tiempo, signal2 + 5, pen=(0,255,0), name = s2)
    if(len(signal3) > 0):
        p1.plot(tiempo, signal3, pen=(0,0,255), name = s3)
    p1.setMouseEnabled(x=True, y=False)
    p1.enableAutoRange(x=False, y=True)
    lr = pg.LinearRegionItem([0, max(tiempo)/3])
    lr.setZValue(-10)
    p1.addItem(lr)
    win.nextRow()

    p2 = win.addPlot(title=s1)
    p2.plot(tiempo, signal1, pen=(255,0,0))
    p2.setDownsampling(auto=False, ds=4, mode='mean')
    p2.showGrid(x = True, y = True, alpha = 0.3)     
    def updatePlot():
        p2.setXRange(*lr.getRegion(), padding=0)
    def updateRegion():
        lr.setRegion(p2.getViewBox().viewRange()[0])
    lr.sigRegionChanged.connect(updatePlot)
    p2.sigXRangeChanged.connect(updateRegion)
    p2.setMouseEnabled(x=True, y=True)
    p2.enableAutoRange(x=False, y=True)
    updatePlot()

    win.nextRow()

    if(len(signal2) > 0):
        p3 = win.addPlot(title=s2)
        p3.plot(tiempo, signal2, pen=(0,255,0))
        p3.setDownsampling(auto=False, ds=4, mode='mean')
        p3.showGrid(x = True, y = True, alpha = 0.3)     
        p3.setXLink(p2)
        p3.setYLink(p2)
        p3.setMouseEnabled(x=True, y=False)
        p3.enableAutoRange(x=False, y=False)
        lr.sigRegionChanged.connect(updatePlot)
        p3.sigXRangeChanged.connect(updateRegion)
        updatePlot()

        win.nextRow()
        if(len(signal3) > 0):
            p4 = win.addPlot(title= s3)
            p4.plot(tiempo, signal3, pen=(0,0,255))
            p4.setDownsampling(auto=False, ds=4, mode='mean')
            p4.showGrid(x = True, y = True, alpha = 0.3)     
            p4.setXLink(p2)
            p4.setYLink(p2)
            p4.setMouseEnabled(x=True, y=True)
            p4.enableAutoRange(x=False, y=False)
            lr.sigRegionChanged.connect(updatePlot)
            p4.sigXRangeChanged.connect(updateRegion)
            updatePlot()

    for idx, evento in enumerate(annotations):
        for anotacion in annotations[evento]:
            region1 = pg.LinearRegionItem([anotacion[0],(anotacion[0] + anotacion[1])], movable=False, pen = colors[idx][1], brush = colors[idx][0])
            region1.setZValue(-10)
            p1.addItem(region1)
            region2 = pg.LinearRegionItem([anotacion[0],(anotacion[0] + anotacion[1])], movable=False, pen = colors[idx][1], brush = colors[idx][0])
            region2.setZValue(-10)
            p2.addItem(region2)
            if(len(signal2) > 0):
                region3 = pg.LinearRegionItem([anotacion[0],(anotacion[0] + anotacion[1])], movable=False, pen = colors[idx][1], brush = colors[idx][0])
                region3.setZValue(-10)
                p3.addItem(region3)
            if(len(signal3) > 0):
                region4 = pg.LinearRegionItem([anotacion[0],(anotacion[0] + anotacion[1])], movable=False, pen = colors[idx][1], brush = colors[idx][0])
                region4.setZValue(-10)
                p4.addItem(region4)
    pg.exec()




archivo = 4 #CHANGE
path_edf = "" #CHANGE
path_annot = "" #CHANGE

all_signals = read_signals_EDF(path_edf)
annotations = get_annotations(path_annot)

bipolar_signal, tiempo, sampling = get_bipolar_signal(all_signals['C3'], all_signals['O1'])

plot_signals(annotations, tiempo, all_signals['C3']['Signal'], 'C3', all_signals['O1']['Signal'], 'O1', bipolar_signal, 'C3-O1')