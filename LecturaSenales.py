from Imports import *

def LecturaSenalPandasDF():

    dfCON = pd.read_csv(r'SenalesCONapnea.csv')
    tiempo = dfCON.pop('tiempo')
    dfCON = dfCON.transpose()
    print(dfCON.shape)
    dfSIN = pd.read_csv(r'SenalesSINapnea.csv')
    dfSIN.drop(columns=dfSIN.columns[0], inplace=True)
    dfSIN = dfSIN.transpose()
    print(dfSIN.shape)

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

    dfFinal = dfFinal.sample(frac=1.0).reset_index(drop=True) #shuffle

    return dfFinal, tiempo

def LecturaSenalTensor():
    
    dfCON = pd.read_csv(r'SenalesCONapnea.csv')
    dfCON.drop(columns=dfCON.columns[0], inplace=True)
    print('Cantidad de señales con apnea: ', dfCON.shape[1], ' de ', dfCON.shape[0], ' puntos')
    dfCON = dfCON.transpose()

    dfSIN = pd.read_csv(r'SenalesSINapnea.csv')
    dfSIN.drop(columns=dfSIN.columns[0], inplace=True)
    print('Cantidad de señales sin apnea: ', dfSIN.shape[1], ' de ', dfSIN.shape[0], ' puntos')
    dfSIN = dfSIN.transpose()

    X = np.concatenate((dfCON, dfSIN))
    targetCON = np.ones(dfCON.shape[0])
    targetSIN = np.zeros(dfSIN.shape[0])
    y = np.concatenate((targetCON, targetSIN))

    return X, y

def analisis_datos(trainset, valset, testset):
    cant_datos = len(trainset)
    con_apnea = int(sum(sum((trainset[:][1]).tolist(), [])))
    text = '\nCantidad de datos de entrenamiento: ' + str(cant_datos) + '\t' + 'Con apnea: ' + str(con_apnea) + '\t\t' + 'Sin apnea: ' + str(cant_datos-con_apnea)
    cant_datos = len(valset)
    con_apnea = int(sum(sum((valset[:][1]).tolist(), [])))
    text += '\nCantidad de datos de validacion: ' + str(cant_datos) + '\t\t' + 'Con apnea: ' + str(con_apnea) + '\t\t' + 'Sin apnea: ' + str(cant_datos-con_apnea)
    cant_datos = len(testset)
    con_apnea = int(sum(sum((testset[:][1]).tolist(), [])))
    text += '\nCantidad de datos de prueba: ' + str(cant_datos) + '\t\t' + 'Con apnea: ' + str(con_apnea) + '\t\t' + 'Sin apnea: ' + str(cant_datos-con_apnea) + '\n'
    
    return text



