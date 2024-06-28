try:
    from Imports import *
except:
    from src.Imports import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class ApneaDataset(Dataset):
    """Dataset custom class"""

    def __init__(self, X, y, archivos):
        """
        Initializes the Apnea Dataset.

        Args:
            COMPLETAR

        Returns: none.
        """
        self.__X = torch.tensor(X)
        self.__y = torch.tensor(y)
        self.trainset = []
        self.valset = []
        self.testset = []
        self.archivos = archivos

    def __len__(self):
        return len(self.__y)

    def __getitem__(self, idx:int):
        return self.__X[idx], self.__y[idx]
    
    def signal_len(self):
        return self.__X.shape[2]

    def split_dataset(self, train_perc:float, val_perc:float, test_perc:float = 0.0):
        """
        Split the ApneaDataset into train, validation and test Subsets and save them as attrbiutes of the ApneaDataset object.

        Note: train_perc + val_perc + test_perc must be = 1, otherwise test_perc will be automatically calculated to ensure this condition is satisfied.
        
        Args: 
        - train_perc (float): percentage of training data (0 < train_perc < 1).
        - val_perc (float): percentage of validation data (0 < val_perc < 1).
        - test_perc (float): percentage of test data (0 < test_perc < 1).

        Returns: none.
        """

        if((train_perc + val_perc + test_perc) != 1.0):
            test_perc = 1.0 - (train_perc + val_perc)
        if((train_perc + val_perc + test_perc) > 1.0):
            raise Exception("La suma de los porcentajes no puede superar 1.0")


        labels = np.array([self[i][1].item() for i in range(len(self))])
        
        train_val_indices, test_indices = train_test_split(
            np.arange(len(labels)),
            test_size=test_perc,
            stratify=labels
        )
        
        train_indices, val_indices = train_test_split(
            train_val_indices,
            test_size=val_perc / (train_perc + val_perc),
            stratify=labels[train_val_indices]
        )
        
        self.trainset = Subset(self, train_indices)
        self.valset = Subset(self, val_indices)
        self.testset = Subset(self, test_indices)
    
    def analisis_datos(self):
        """
        Calculates and returns a string containing the analysis of the dataset, including the count of data for training, validation, and testing sets, as well as the count of data with and without apnea in each set.
        
        Args: none.

        Returns: 
        - string containing the analysis of the dataset. 
        """

        cant_datos = len(self.trainset)
        con_apnea = int(sum(sum((self.trainset[:][1]).tolist(), [])))
        text = '\nTraining data count: ' + str(cant_datos) + '\n\t' + 'With apnea: ' + str(con_apnea) + '\n\t' + 'Without apnea: ' + str(cant_datos-con_apnea)
        cant_datos = len(self.valset)
        con_apnea = int(sum(sum((self.valset[:][1]).tolist(), [])))
        text += '\nValidation data count: ' + str(cant_datos) + '\n\t' + 'With apnea: ' + str(con_apnea) + '\n\t' + 'Without apnea: ' + str(cant_datos-con_apnea)
        cant_datos = len(self.testset)
        con_apnea = int(sum(sum((self.testset[:][1]).tolist(), [])))
        text += '\nTest data count: ' + str(cant_datos) + '\n\t' + 'With apnea: ' + str(con_apnea) + '\n\t' + 'Without apnea: ' + str(cant_datos-con_apnea) + '\n'
        
        return text

    def undersample_majority_class(self, majority_class, val = True):
        
        class_counts_train = Counter([int(self.trainset[i][1].item()) for i in range(len(self.trainset))])

        min_count_train = min(class_counts_train.values())
        
        indices_train = []

        class_counter_train = {cls: 0 for cls in class_counts_train}

        for idx in range(len(self.trainset)):
            label = int(self.trainset[idx][1].item())
            if label == majority_class:
                if class_counter_train[label] < min_count_train:
                    indices_train.append(idx)
                    class_counter_train[label] += 1
            else:
                indices_train.append(idx)
        
        self.trainset = Subset(self.trainset, indices_train)
        
        if(val):
            class_counts_val = Counter([int(self.valset[i][1].item()) for i in range(len(self.valset))])

            min_count_val = min(class_counts_val.values())
            
            indices_val = []

            class_counter_val = {cls: 0 for cls in class_counts_val}

            for idx in range(len(self.valset)):
                label = int(self.valset[idx][1].item())
                if label == majority_class:
                    if class_counter_val[label] < min_count_val:
                        indices_val.append(idx)
                        class_counter_val[label] += 1
                else:
                    indices_val.append(idx)
            
            self.valset = Subset(self.valset, indices_val)
    
    @staticmethod
    def standarize_data(X):
        scaler = StandardScaler()
        scaler.fit(X)
        return scaler.transform(X)

    @staticmethod
    def from_csv(csv_path_con:str, csv_path_sin:str):
        """
        Args:
            csv_path_con (str): path to the 'SenalesCONapnea.csv' file.
            csv_path_sin (str): path to the 'SenalesSINapnea.csv' file.

        Returns: none.
        """

        #Read .csv files and drop time column
        try:
            dfCON = pd.read_csv(csv_path_con)
        except FileNotFoundError:
            raise FileNotFoundError("File not found. Path should look like 'data\ApneaDetection_SimulatedSignals\SenalesCONapnea.csv'")
        dfCON.drop(columns=dfCON.columns[0], inplace=True)
        dfCON = dfCON.transpose()
        try:
            dfSIN = pd.read_csv(csv_path_sin)
        except FileNotFoundError:
            raise FileNotFoundError("File not found. Path should look like 'data\ApneaDetection_SimulatedSignals\SenalesSINapnea.csv'")
        dfSIN.drop(columns=dfSIN.columns[0], inplace=True)
        dfSIN = dfSIN.transpose()

        #Label and join dataframes 
        X = np.concatenate((dfCON, dfSIN))
        targetCON = np.ones(dfCON.shape[0])
        targetSIN = np.zeros(dfSIN.shape[0])
        y = np.concatenate((targetCON, targetSIN))

        X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
        return X,y

    @staticmethod
    def from_segments(segments, stand = False):
        X = []
        y = []

        for segment in segments:
            X.append(segment['Signal'])
            y.append(segment['Label'])

        # Convertimos las listas en arreglos numpy
        X = np.vstack(X)
        y = np.array(y)
        
        if(stand):
            X = ApneaDataset.standarize_data(X)

        X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

        return X,y