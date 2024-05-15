import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def LecturaSenalPandasDF():
    dfCON = pd.read_csv(r'SenalesCONapnea.csv')
    tiempo = dfCON.pop('tiempo')
    dfCON = dfCON.transpose()
    print(dfCON.shape)
    dfSIN = pd.read_csv(r'SenalesSINapnea.csv')
    dfSIN.drop(columns=dfSIN.columns[0], inplace=True)
    dfSIN = dfSIN.transpose()
    #print(dfSIN.shape)

    targetCON = np.ones(dfCON.shape[0])
    targetSIN = np.zeros(dfSIN.shape[0])
    target = np.concatenate((targetCON, targetSIN))
    dfTarget = pd.DataFrame(target)

    df = pd.concat([dfCON, dfSIN])

    lista_df = df.values.tolist()
    lista_dfTarget = dfTarget.values.tolist()
    for element in lista_df:
        element.append(lista_dfTarget[lista_df.index(element)][0])

    dfFinal = pd.DataFrame(lista_df)
    #print(dfFinal.head)
    #print(dfFinal.tail)

    dfFinal = dfFinal.sample(frac=1.0).reset_index(drop=True) #shuffle
    #print(dfFinal.head)

    return dfFinal, tiempo


def LecturaSenalTensor():
    
    dfCON = pd.read_csv(r'SenalesCONapnea.csv')
    dfCON.drop(columns=dfCON.columns[0], inplace=True)
    dfCON = dfCON.transpose()
    dfSIN = pd.read_csv(r'SenalesSINapnea.csv')
    dfSIN.drop(columns=dfSIN.columns[0], inplace=True)
    dfSIN = dfSIN.transpose()

    X = np.concatenate((dfCON, dfSIN))
    targetCON = np.ones(dfCON.shape[0])
    targetSIN = np.zeros(dfSIN.shape[0])
    y = np.concatenate((targetCON, targetSIN))

    return X, y


#colores = ['blue', 'red']
#for index, row in datos.iterrows():
#    eeg_signal = row[:-1]
#    target = int(row.iloc[-1])
#    titulo = f'Señal EEG - Target: {target}'
    
#    plt.figure()
#    plt.plot(tiempo, eeg_signal, color=colores[target])
#    plt.title(titulo)
#    plt.xlabel('Tiempo')
#    plt.ylabel('Amplitud')
#    plt.ylim(-3, 3)
#    plt.show()
