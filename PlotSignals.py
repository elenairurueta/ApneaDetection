import pyqtgraph.examples
import numpy as np
import pyqtgraph as pg
from LecturaSenalesReales import read_signals_EDF
from LecturaAnotaciones import Anotaciones
from PyQt5.QtGui import QBrush, QColor

all_signals = read_signals_EDF('C:\\Users\\elena\\OneDrive\\Documentos\\Tesis\\Dataset\\HomePAP\\polysomnography\\edfs\\lab\\full\\homepap-lab-full-1600024.edf')
annotations = Anotaciones('C:\\Users\\elena\\OneDrive\\Documentos\\Tesis\\Dataset\\HomePAP\\polysomnography\\annotations-events-profusion\\lab\\full\\homepap-lab-full-1600024-profusion.xml')

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

s1 = 'C4'
s2 = 'M1'

x = all_signals[s1]['Time']
data1 = all_signals[s1]['Signal']
data2 = all_signals[s2]['Signal']
data3 = all_signals[s1]['Signal'] - all_signals[s2]['Signal']

p1 = win.addPlot(title="Region Selection")
p1.addLegend()
p1.setDownsampling(auto=False, ds=4, mode='mean')
p1.showGrid(x = True, y = True, alpha = 0.3)     
p1.plot(x, data1 + 10, pen=(255,0,0), name = s1)
p1.plot(x, data2 + 5, pen=(0,255,0), name = s2)
p1.plot(x, data3, pen=(0,0,255), name = s1 + '-' + s2)
p1.setMouseEnabled(x=True, y=False)
p1.enableAutoRange(x=False, y=True)
lr = pg.LinearRegionItem([0, max(x)/3])
lr.setZValue(-10)
p1.addItem(lr)
win.nextRow()

p2 = win.addPlot(title=s1)
p2.plot(x, data1, pen=(255,0,0))
p2.setDownsampling(auto=False, ds=4, mode='mean')
p2.showGrid(x = True, y = True, alpha = 0.3)     
def updatePlot():
    p2.setXRange(*lr.getRegion(), padding=0)
def updateRegion():
    lr.setRegion(p2.getViewBox().viewRange()[0])
lr.sigRegionChanged.connect(updatePlot)
p2.sigXRangeChanged.connect(updateRegion)
p2.setMouseEnabled(x=True, y=False)
p2.enableAutoRange(x=False, y=True)
updatePlot()


win.nextRow()

p3 = win.addPlot(title=s2)
p3.plot(x, data2, pen=(0,255,0))
p3.setDownsampling(auto=False, ds=4, mode='mean')
p3.showGrid(x = True, y = True, alpha = 0.3)     
p3.setXLink(p2)
p3.setMouseEnabled(x=True, y=False)
p3.enableAutoRange(x=False, y=True)
lr.sigRegionChanged.connect(updatePlot)
p3.sigXRangeChanged.connect(updateRegion)
updatePlot()

win.nextRow()

p4 = win.addPlot(title= s1 + '-' + s2)
p4.plot(x, data3, pen=(0,0,255))
p4.setDownsampling(auto=False, ds=4, mode='mean')
p4.showGrid(x = True, y = True, alpha = 0.3)     
p4.setXLink(p2)
p4.setMouseEnabled(x=True, y=False)
p4.enableAutoRange(x=False, y=True)
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
        region3 = pg.LinearRegionItem([anotacion[0],(anotacion[0] + anotacion[1])], movable=False, pen = colors[idx][1], brush = colors[idx][0])
        region3.setZValue(-10)
        p3.addItem(region3)
        region4 = pg.LinearRegionItem([anotacion[0],(anotacion[0] + anotacion[1])], movable=False, pen = colors[idx][1], brush = colors[idx][0])
        region4.setZValue(-10)
        p4.addItem(region4)

if __name__ == '__main__':
    pg.exec()
