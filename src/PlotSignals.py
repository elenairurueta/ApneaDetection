import pyqtgraph.examples
import numpy as np
import pyqtgraph as pg
from LecturaSenalesReales import read_signals_EDF
from LecturaAnotaciones import Anotaciones
from PyQt5.QtGui import QBrush, QColor


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
    p2.setMouseEnabled(x=True, y=False)
    p2.enableAutoRange(x=False, y=True)
    updatePlot()

    win.nextRow()

    if(len(signal2) > 0):
        p3 = win.addPlot(title=s2)
        p3.plot(tiempo, signal2, pen=(0,255,0))
        p3.setDownsampling(auto=False, ds=4, mode='mean')
        p3.showGrid(x = True, y = True, alpha = 0.3)     
        p3.setXLink(p2)
        p3.setMouseEnabled(x=True, y=False)
        p3.enableAutoRange(x=False, y=True)
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
            if(len(signal2) > 0):
                region3 = pg.LinearRegionItem([anotacion[0],(anotacion[0] + anotacion[1])], movable=False, pen = colors[idx][1], brush = colors[idx][0])
                region3.setZValue(-10)
                p3.addItem(region3)
            if(len(signal3) > 0):
                region4 = pg.LinearRegionItem([anotacion[0],(anotacion[0] + anotacion[1])], movable=False, pen = colors[idx][1], brush = colors[idx][0])
                region4.setZValue(-10)
                p4.addItem(region4)
    pg.exec()

