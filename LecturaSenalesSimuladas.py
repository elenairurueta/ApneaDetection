from Imports import *

class ApneaDataset(Dataset):

    def __init__(self, csv_path_con, csv_path_sin):
        dfCON = pd.read_csv(csv_path_con)
        dfCON.drop(columns=dfCON.columns[0], inplace=True)
        dfCON = dfCON.transpose()

        dfSIN = pd.read_csv(csv_path_sin)
        dfSIN.drop(columns=dfSIN.columns[0], inplace=True)
        dfSIN = dfSIN.transpose()

        self.__X = np.concatenate((dfCON, dfSIN))
        targetCON = np.ones(dfCON.shape[0])
        targetSIN = np.zeros(dfSIN.shape[0])
        self.__y = np.concatenate((targetCON, targetSIN))

        self.__X = torch.tensor(self.__X, dtype=torch.float32).unsqueeze(1)
        self.__y = torch.tensor(self.__y, dtype=torch.float32).reshape(-1, 1)

        self.trainset = []
        self.valset = []
        self.testset = []

    def __len__(self):
        return len(self.__y)

    def __getitem__(self, idx):
        return self.__X[idx], self.__y[idx]
    
    def signal_len(self):
        return self.__X.shape[2]

    def split_dataset(self, train_perc, val_perc, test_perc = 0.0):
        if((test_perc == 0.0) or ((train_perc + val_perc + test_perc) < 1.0)):
            test_perc = 1.0 - (train_perc + val_perc)
        if((train_perc + val_perc + test_perc) > 1.0):
            raise Exception("La suma de los porcentajes no puede superar 1.0")
        train_size = int(train_perc * self.__len__())  # 60% para entrenamiento
        val_size = int(val_perc * self.__len__())    # 20% para validación
        test_size = int(test_perc * self.__len__())  # 20% para prueba
        self.trainset, self.valset, self.testset = random_split(self, [train_size, val_size, test_size])
    
    def analisis_datos(self):
        cant_datos = len(self.trainset)
        con_apnea = int(sum(sum((self.trainset[:][1]).tolist(), [])))
        text = '\nCantidad de datos de entrenamiento: ' + str(cant_datos) + '\n\t' + 'Con apnea: ' + str(con_apnea) + '\n\t' + 'Sin apnea: ' + str(cant_datos-con_apnea)
        cant_datos = len(self.valset)
        con_apnea = int(sum(sum((self.valset[:][1]).tolist(), [])))
        text += '\nCantidad de datos de validacion: ' + str(cant_datos) + '\n\t' + 'Con apnea: ' + str(con_apnea) + '\n\t' + 'Sin apnea: ' + str(cant_datos-con_apnea)
        cant_datos = len(self.testset)
        con_apnea = int(sum(sum((self.testset[:][1]).tolist(), [])))
        text += '\nCantidad de datos de prueba: ' + str(cant_datos) + '\n\t' + 'Con apnea: ' + str(con_apnea) + '\n\t' + 'Sin apnea: ' + str(cant_datos-con_apnea) + '\n'
        
        return text