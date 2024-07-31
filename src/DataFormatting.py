# try:
#     from Imports import *
#     from LecturaSenalesReales import *
#     from LecturaAnotaciones import *
# except:
#     from src.Imports import *
#     from src.LecturaSenalesReales import *
#     from src.LecturaAnotaciones import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.signal import resample
import copy
from Imports import *
from LecturaSenalesReales import *
from LecturaAnotaciones import *

class ApneaDataset(Dataset):
    """Dataset custom class"""

    def __init__(self, X, y, archivos):
        """
        Initializes the Apnea Dataset.

        Args:

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
                    indices_train.append(self.trainset.indices[idx])
                    class_counter_train[label] += 1
            else:
                indices_train.append(self.trainset.indices[idx])
        
        self.trainset = Subset(self.trainset.dataset, indices_train)
        
        if(val):
            class_counts_val = Counter([int(self.valset[i][1].item()) for i in range(len(self.valset))])

            min_count_val = min(class_counts_val.values())
            
            indices_val = []

            class_counter_val = {cls: 0 for cls in class_counts_val}

            for idx in range(len(self.valset)):
                label = int(self.valset[idx][1].item())
                if label == majority_class:
                    if class_counter_val[label] < min_count_val:
                        indices_val.append(self.valset.indices[idx])
                        class_counter_val[label] += 1
                else:
                    indices_val.append(self.valset.indices[idx])
            
            self.valset = Subset(self.valset.dataset, indices_val)
    
    def save_dataset(self, file_path:str):
        """
        Save the ApneaDataset and its subsets to a file.
        
        Args:
            apnea_dataset (ApneaDataset): the dataset to save.
            file_path (str): the path where the dataset will be saved.
            
        Returns:
            None
        """
        data = {
            'X': self.__X,
            'y': self.__y,
            'archivos': self.archivos,
            'trainset': self.trainset if isinstance(self.trainset, Subset) else [],
            'valset': self.valset if isinstance(self.valset, Subset) else [],
            'testset': self.testset if isinstance(self.testset, Subset) else []
        }

        torch.save(data, file_path)

    @staticmethod
    def load_dataset(file_path:str):
        """
        Load the ApneaDataset and its subsets from a file.
        
        Args:
            file_path (str): the path where the dataset is saved.
            
        Returns:
            ApneaDataset: the loaded dataset.
        """
        data = torch.load(file_path)
        apnea_dataset = ApneaDataset(data['X'], data['y'], data['archivos'])
        if len(data['trainset'])>0:
            apnea_dataset.trainset = data['trainset']
        if len(data['valset'])>0:
            apnea_dataset.valset = data['valset']
        if len(data['testset'])>0:
            apnea_dataset.testset = data['testset']
        
        analisis_datos = apnea_dataset.analisis_datos()
        print(analisis_datos)
        return apnea_dataset

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

    @staticmethod
    def create_datasets(archivos):
        for archivo in archivos:
            path_edf = f"C:\\Users\\elena\\OneDrive\\Documentos\\Tesis\\Dataset\\HomePAP\\polysomnography\\edfs\\lab\\full\\homepap-lab-full-1600{archivo:03d}.edf"
            path_annot = f"C:\\Users\\elena\\OneDrive\\Documentos\\Tesis\\Dataset\\HomePAP\\polysomnography\\annotations-events-profusion\\lab\\full\\homepap-lab-full-1600{archivo:03d}-profusion.xml"
            all_signals = read_signals_EDF(path_edf)
            annotations = Anotaciones(path_annot)

            bipolar_signal, tiempo, sampling = get_bipolar_signal(all_signals['C3'], all_signals['O1'])

            segments = get_signal_segments_strict(bipolar_signal, tiempo, sampling, annotations)

            X,y = ApneaDataset.from_segments(segments, stand = True)
            dataset = ApneaDataset(X,y, archivo)

            dataset.split_dataset(train_perc = 0.8, 
                                val_perc = 0.1, 
                                test_perc = 0.1)

            dataset.undersample_majority_class(0.0)

            dataset.save_dataset(f"data\ApneaDetection_HomePAPSignals\datasets\dataset_archivo_1600{archivo:03d}.pth")


class ApneaDataset2(Dataset):
    """Dataset custom class"""

    def __init__(self, X, y, sr, archivos):
        """
        Initializes the Apnea Dataset.

        Args:

        Returns: none.
        """
        self.__X = X.clone().detach()
        self.__y = y.clone().detach()
        self.subsets = []
        self.archivos = archivos
        self.__sr = sr

    def __len__(self):
        return len(self.__y)

    def __getitem__(self, idx:int):
        return self.__X[idx], self.__y[idx]
    
    def signal_len(self):
        return self.__X.shape[2]

    def split_dataset(self):

        labels = np.array([self[i][1].item() for i in range(len(self))])
        
        other_indices = np.arange(len(labels))
        for i in range(0,9):
            other_indices, set_indices = train_test_split(
                other_indices,
                test_size=round(0.1/(1-0.1*i), 2),
                stratify=labels[other_indices]
            )
            self.subsets.append(Subset(self, set_indices))
        self.subsets.append(Subset(self, other_indices))

    def get_subsets(self, idxs):
        indices = []
        for idx in idxs:
            subset_indices = list(self.subsets[idx].indices)
            indices.extend(subset_indices)
        return Subset(self.subsets[idx].dataset, indices)
    
    def analisis_datos(self):
        """
        Calculates and returns a string containing the analysis of the dataset, including the count of data for training, validation, and testing sets, as well as the count of data with and without apnea in each set.
        
        Args: none.

        Returns: 
        - string containing the analysis of the dataset. 
        """
        text = ""
        for idx, subset in enumerate(self.subsets):
            cant_datos = len(subset)
            con_apnea = int(sum(sum((subset[:][1]).tolist(), [])))
            text += f'\nSubset {idx} data count: ' + str(cant_datos) + '\n\t' + 'With apnea: ' + str(con_apnea) + '\n\t' + 'Without apnea: ' + str(cant_datos-con_apnea)
        return text

    def undersample_majority_class(self, majority_class, subsets, prop:float = 1):
        """ Si prop = 1, maxclass = minclass
            Si prop = 2, maxclass = 2*minclass
        """
        for subset in subsets:
            class_counts = Counter([int(self.subsets[subset][i][1].item()) for i in range(len(self.subsets[subset]))])

            min_count = min(class_counts.values())
            
            indices = []

            class_counter = {cls: 0 for cls in class_counts}

            for idx in range(len(self.subsets[subset])):
                label = int(self.subsets[subset][idx][1].item())
                if label == majority_class:
                    if class_counter[label] < min_count*prop:
                        indices.append(self.subsets[subset].indices[idx])
                        class_counter[label] += 1
                else:
                    indices.append(self.subsets[subset].indices[idx])
            
            self.subsets[subset] = Subset(self.subsets[subset].dataset, indices)
    
    def save_dataset(self, file_path:str):
        """
        Save the ApneaDataset and its subsets to a file.
        
        Args:
            apnea_dataset (ApneaDataset): the dataset to save.
            file_path (str): the path where the dataset will be saved.
            
        Returns:
            None
        """
        data = {
            'X': self.__X,
            'y': self.__y,
            'sr': self.__sr,
            'archivos': self.archivos,
            'subsets': self.subsets if len(self.subsets)>0 else []
        }

        torch.save(data, file_path)

    def plot_segments(self, idx = None):
        plt.figure(1)
        if(idx == None):
            for segment in self.__X:
                tiempo_segmento = np.arange(0, 30, 1/self.__sr)
                plt.plot(tiempo_segmento, segment[0])
                plt.show()
        else:
            tiempo_segmento = np.arange(0, 30, 1/self.__sr)
            plt.plot(tiempo_segmento, self.__X[idx][0])
            plt.show()

    def resample_segments(self, sampling_rate):
        new_X = []
        long_segmento = 30
        num_puntos = long_segmento * sampling_rate
        for segment in self.__X:
            new_segment = [resample(segment[0].numpy(), num_puntos)]
            new_X.append(new_segment)
        self.__X = torch.tensor(new_X)
        self.__sr = sampling_rate
        for idx, subset in enumerate(self.subsets):
            self.subsets[idx].dataset = self

    # @staticmethod
    # def join_datasets(datasets, traintestval = None):

    #     if not isinstance(datasets[0], ApneaDataset2):
    #             raise TypeError("Expected an instance of ApneaDataset2")
    #     appended_dataset = copy.deepcopy(datasets[0])
    #     if(traintestval != None):
    #         train_subsets, val_subsets, test_subsets = traintestval[0]
    #     if(len(datasets) > 1):
    #         for idx, dataset in enumerate(datasets[1:]):
    #             if not isinstance(dataset, ApneaDataset2):
    #                 raise TypeError("Expected an instance of ApneaDataset2")
    #             appended_dataset.__X = torch.cat((appended_dataset.__X, dataset.__X), dim=0)
    #             appended_dataset.__y = torch.cat((torch.tensor(appended_dataset.__y), torch.tensor(dataset.__y)), dim=0)

    #             if(traintestval != None):
    #                 train_subsets_viejo, val_subsets_viejo, test_subsets_viejo = traintestval[idx + 1]
    #                 train_subsets += [x+len(appended_dataset.subsets) for x in train_subsets_viejo]
    #                 val_subsets += [x+len(appended_dataset.subsets) for x in val_subsets_viejo]
    #                 test_subsets += [x+len(appended_dataset.subsets) for x in test_subsets_viejo]
                
    #             appended_dataset.subsets += dataset.subsets
                
    #             try:
    #                 appended_dataset.archivos.append(dataset.archivos)
    #             except:
    #                 appended_dataset.archivos = [appended_dataset.archivos] + [dataset.archivos]
    #                 print(appended_dataset.archivos)

    #     return appended_dataset, train_subsets, val_subsets, test_subsets
    
    @staticmethod
    def join_datasets(datasets, traintestval=None):
        if not isinstance(datasets[0], ApneaDataset2):
            raise TypeError("Expected an instance of ApneaDataset2")

        appended_dataset = copy.deepcopy(datasets[0])
        new_subsets = []
        offset = 0

        if traintestval is not None:
            train_subsets, val_subsets, test_subsets = traintestval[0]

        for idx, dataset in enumerate(datasets):
            if not isinstance(dataset, ApneaDataset2):
                raise TypeError("Expected an instance of ApneaDataset2")

            if idx > 0:  # Skip the first dataset since it's already copied
                appended_dataset.__X = torch.cat((appended_dataset.__X, dataset.__X), dim=0)
                appended_dataset.__y = torch.cat((appended_dataset.__y, dataset.__y.clone().detach()), dim=0)
                try:
                    appended_dataset.archivos.append(dataset.archivos)
                except AttributeError:
                    appended_dataset.archivos = [appended_dataset.archivos] + [dataset.archivos]

            if(traintestval != None and idx > 0):
                train_subsets_viejo, val_subsets_viejo, test_subsets_viejo = traintestval[idx]
                train_subsets += [x+len(new_subsets) for x in train_subsets_viejo]
                val_subsets += [x+len(new_subsets) for x in val_subsets_viejo]
                test_subsets += [x+len(new_subsets) for x in test_subsets_viejo]

            for subset in dataset.subsets:
                new_indices = [i + offset for i in subset.indices]
                new_subsets.append(Subset(appended_dataset, new_indices))

            offset += len(dataset.__X)

            

        appended_dataset.subsets = new_subsets

        return appended_dataset, train_subsets, val_subsets, test_subsets
    
    @staticmethod
    def load_dataset(file_path:str):
        """
        Load the ApneaDataset and its subsets from a file.
        
        Args:
            file_path (str): the path where the dataset is saved.
            
        Returns:
            ApneaDataset: the loaded dataset.
        """
        data = torch.load(file_path)
        apnea_dataset = ApneaDataset2(data['X'], data['y'], data['sr'], data['archivos'])
        if len(data['subsets'])>0:
            apnea_dataset.subsets = data['subsets']
        return apnea_dataset

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

    @staticmethod
    def create_datasets(archivos):

        for archivo in archivos:
            path_edf = f"C:\\Users\\elena\\OneDrive\\Documentos\\Tesis\\Dataset\\HomePAP\\polysomnography\\edfs\\lab\\full\\homepap-lab-full-1600{archivo:03d}.edf"
            path_annot = f"C:\\Users\\elena\\OneDrive\\Documentos\\Tesis\\Dataset\\HomePAP\\polysomnography\\annotations-events-profusion\\lab\\full\\homepap-lab-full-1600{archivo:03d}-profusion.xml"
            all_signals = read_signals_EDF(path_edf)
            annotations = Anotaciones(path_annot)

            bipolar_signal, tiempo, sampling = get_bipolar_signal(all_signals['C3'], all_signals['O1'])

            segments = get_signal_segments_strict(bipolar_signal, tiempo, sampling, annotations)

            X,y = ApneaDataset2.from_segments(segments, stand = True)
            dataset = ApneaDataset2(X, y, sampling, archivo)
            dataset.split_dataset()
            dataset.save_dataset(f"data\ApneaDetection_HomePAPSignals\datasets\dataset2_archivo_1600{archivo:03d}.pth")
