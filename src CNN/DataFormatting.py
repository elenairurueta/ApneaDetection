from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.signal import resample
import copy
from Imports import *
from RealSignals import *
from Annotations import get_annotations
from Filtering import filter_Butter, filter_FIR2, filter_Notch_FIR

class ApneaDataset(Dataset):
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
        """
        Splits the dataset into 10 subsets, ensuring that each subset contains a balanced distribution of classes.
        It modifies the `self.subsets` attribute by appending 10 subsets.

        Returns: none
        """

        labels = np.array([self[i][1].item() for i in range(len(self))]) #Extract the labels from the dataset
       
        other_indices = np.arange(len(labels)) #Create an array of indices corresponding to the dataset
        for i in range(0,9): #Get 9 stratified splits, leaving the last 10% for the final subset
            other_indices, set_indices = train_test_split(
                other_indices,
                test_size=round(0.1/(1-0.1*i), 2), #Calculate test size to ensure 10% of the remaining data is split off
                stratify=labels[other_indices] #Stratified split to maintain class balance
            )
            self.subsets.append(Subset(self, set_indices))
        self.subsets.append(Subset(self, other_indices))
   
    def get_subsets(self, idxs):
        indices = []
        for idx in idxs:
            subset_indices = list(self.subsets[idx].indices)
            indices.extend(subset_indices)
        return Subset(self.subsets[idx].dataset, indices)

    def data_analysis(self):
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
        """
        Undersamples the majority class in specified subsets to achieve a balanced class distribution. It modifies `self.subsets` in place.

        Parameters:
        - majority_class: The label of the majority class that needs to be undersampled.
        - subsets: A list of indices specifying which subsets to process.
        - prop (float): Proportion of the minimum class count to which the majority class will be undersampled (default is 1, meaning the majority class is reduced to the size of the smallest class).

        Returns: none
        """
        for subset in subsets:
            class_counts = Counter([int(self.subsets[subset][i][1].item()) for i in range(len(self.subsets[subset]))]) #Count the number of samples for each class in the subset
            min_count = min(class_counts.values()) #Determine the minimum number of samples in any class
            indices = []
            class_counter = {cls: 0 for cls in class_counts}

            for idx in range(len(self.subsets[subset])):
                label = int(self.subsets[subset][idx][1].item())
                if label == majority_class:
                    #If the sample is from the majority class and we haven't reached the proportional limit, include it
                    if class_counter[label] < min_count*prop:
                        indices.append(self.subsets[subset].indices[idx])
                        class_counter[label] += 1
                else:
                    #Include all samples from other classes
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
        """
        Resamples the signal segments to a uniform length based on the specified sampling rate. It modifies `self.__X` and updates the subsets' datasets in place.

        Parameters:
        - sampling_rate: The desired sampling rate (in Hz) to resample the segments.

        Returns: none
        """
        new_X = []
        segment_length = 30
        num_puntos = segment_length * sampling_rate
        if(self.__X.shape[1] == 1):
            for segment in self.__X:
                new_segment = [resample(segment[0].numpy(), num_puntos)]
                new_X.append(new_segment)
        elif(self.__X.shape[1] == 2):
            for signal in self.__X:
                new_signal = []
                for segment in signal:
                    new_segment = resample(segment.numpy(), num_puntos)
                    new_signal.append(new_segment)
                new_X.append(new_signal)
        else:
            for segment in self.__X:
                new_segment = resample(segment, num_puntos)
                new_X.append(new_segment)
        new_X_np = np.array(new_X)
        self.__X = torch.tensor(new_X_np)
        for idx, subset in enumerate(self.subsets):
            self.subsets[idx].dataset = self


    @staticmethod
    def join_datasets(datasets, traintestval=None):
        """
        Merges multiple instances of `ApneaDataset2` into a single dataset, combining their data and subsets.

        Parameters:
        - datasets: A list of `ApneaDataset2` instances to be joined.
        - traintestval (tuple of lists, optional): Contains train, validation, and test subsets for each dataset in `datasets`. It should be a list of tuples where each tuple has three lists (train, val, test subsets). Only required if you want to merge the subsets along with the datasets.

        Returns:
        - A tuple containing:
        1. The merged `ApneaDataset2` instance.
        2. Updated train subset indices.
        3. Updated validation subset indices.
        4. Updated test subset indices.
        """

        if not isinstance(datasets[0], ApneaDataset):
            raise TypeError("Expected an instance of ApneaDataset")
        appended_dataset = copy.deepcopy(datasets[0]) #Create a deep copy of the first dataset to serve as the base
        new_subsets = []
        offset = 0
        if traintestval is not None:
            train_subsets, val_subsets, test_subsets = traintestval[0]

        for idx, dataset in enumerate(datasets):
            if not isinstance(dataset, ApneaDataset):
                raise TypeError("Expected an instance of ApneaDataset2")
            if idx > 0:  # Skip the first dataset since it's already copied
                #concatenate their data
                appended_dataset.__X = torch.cat((appended_dataset.__X, dataset.__X), dim=0)
                appended_dataset.__y = torch.cat((appended_dataset.__y, dataset.__y.clone().detach()), dim=0)
                try:
                    appended_dataset.archivos.append(dataset.archivos)
                except AttributeError:
                    appended_dataset.archivos = [appended_dataset.archivos] + [dataset.archivos]

            if(traintestval != None and idx > 0):
                train_subsets_old, val_subsets_old, test_subsets_old = traintestval[idx]
                train_subsets += [x+len(new_subsets) for x in train_subsets_old]
                val_subsets += [x+len(new_subsets) for x in val_subsets_old]
                test_subsets += [x+len(new_subsets) for x in test_subsets_old]

            for subset in dataset.subsets:
                #Adjust indices for the new subsets and add them to the new_subsets list
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
        apnea_dataset = ApneaDataset(data['X'], data['y'], data['sr'], data['archivos'])
        if len(data['subsets'])>0:
            apnea_dataset.subsets = data['subsets']
        return apnea_dataset

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
    def from_segments(segments1, segments2 = None):
        X = []
        y = []

        if(segments2 == None):
            for segment in segments1:
                X.append(segment['Signal'])
                y.append(segment['Label'])
        else:
            if(len(segments1) == len(segments2)):
                for i in range(0, len(segments1)):
                    add = [segments1[i]['Signal'], segments2[i]['Signal']]
                    X.append(add)
                    if(segments1[i]['Label'] == segments2[i]['Label']):
                        y.append(segments1[i]['Label'])
                    else:
                        print("LABELS DISTINTAS")
            else:
                print("LONGITUDES DISTINTAS")

        # Convertimos las listas en arreglos numpy para que sea más rápido
        X = np.array(X)
        y = np.array(y)

        X = torch.tensor(X, dtype=torch.float32)#.unsqueeze(1)
        y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

        return X,y

    @staticmethod
    def create_datasets(files, path_edf, path_annot, overlap, perc_apnea, signal,
                        filtering = False, 
                        filter = "", 
                        split_dataset = True):
        """
        Creates datasets from EDF files and annotations by processing each file, generating segments, and saving them.

        Parameters:
        - files: A list of file identifiers to process.
        - path_edf: The path to the directory containing EDF files.
        - path_annot: The path to the directory containing annotation files.

        Returns: none
        """
        for file in files:
            all_signals = read_signals_EDF(path_edf + f"\\homepap-lab-full-{str(file)}.edf")
            annotations = get_annotations(path_annot + f"\\homepap-lab-full-{str(file)}-profusion.xml")

            try:
                bipolar_signal, time, sampling = get_bipolar_signal(all_signals[signal[0]], all_signals[signal[1]])  

                if(filtering):
                    if(filter == "Butter"):
                        bipolar_signal = filter_Butter(bipolar_signal, sampling, plot = False)
                    elif(filter == "FIR"):
                        bipolar_signal = filter_FIR2(bipolar_signal, sampling, plot = False)
                    elif(filter == "Notch_FIR"):
                        bipolar_signal = filter_Notch_FIR(bipolar_signal, sampling, plot = False)
                        bipolar_signal = filter_FIR2(bipolar_signal, sampling, plot = False)
                    elif(filter == "FIR_Notch"):
                        bipolar_signal = filter_FIR2(bipolar_signal, sampling, plot = False)
                        bipolar_signal = filter_Notch_FIR(bipolar_signal, sampling, plot = False)
                    elif(filter == "Notch"):
                        bipolar_signal = filter_Notch_FIR(bipolar_signal, sampling, plot = False)

                segments = get_signal_segments(bipolar_signal, time, sampling, annotations, period_length=30, overlap=overlap, perc_apnea=perc_apnea, t_descarte = 5*60)
                #segments = get_signal_segments_strict(bipolar_signal, time, sampling, annotations)
                X,y = ApneaDataset.from_segments(segments)
                dataset = ApneaDataset(X, y, sampling, file)
                if split_dataset:
                    dataset.split_dataset()
                dataset.save_dataset(f".\data\ApneaDetection_HomePAPSignals\datasets\dataset_file_{file}_C4M1_nonstrict.pth")

            except:
                continue

    def Zscore_normalization(self):
        """
        Applies Z-score normalization to the input data (self.__X). 
        The normalization scales the features of the data such that they have a mean of 0 and a standard deviation of 1.
        """
        # scaler = StandardScaler()
        # original_shape = self.__X.shape
        # flattened_X = self.__X.reshape(-1, original_shape[-1])
        # scaler.fit(flattened_X)
        # scaled_X = scaler.transform(flattened_X)
        # self.__X = torch.tensor(scaled_X, dtype=torch.float32).view(original_shape)
        for i in range(self.__X.shape[1]):
            scaler = StandardScaler()
            flat_X = []
            for segment in self.__X:
                flat_X.append(segment[i].numpy())
            scaler.fit(flat_X)
            scaled_X = scaler.transform(flat_X)
            for idx in range(len(self.__X)):
                self.__X[idx][i] = torch.tensor(scaled_X[idx])

        # Para comprobar la normalizacion:
        # normalized_X = np.array([segment.numpy() for segment in self.__X])  # Forma [x, 2, 6000]
        # means = normalized_X.mean(axis=(0, 2))  # Media por señal
        # stds = normalized_X.std(axis=(0, 2))   # STD por señal
        # print("Media por señal después de la normalización:", means)
        # print("Desviación estándar por señal después de la normalización:", stds)







class ApneaDataset_SaO2(Dataset):
    """Dataset custom class"""
    def __init__(self, X_EEG, X_SaO2 = None, y = None, sr_EEG = None, sr_SaO2 = None, archivos = None):
        """
        Initializes the Apnea Dataset.
        Args:
        Returns: none.
        """
        self.__X_EEG = X_EEG.clone().detach()
        if(X_SaO2 != None):
            self.__X_SaO2 = X_SaO2.clone().detach()
        self.__y = y.clone().detach()
        self.subsets = []
        self.archivos = archivos
        self.__sr_EEG = sr_EEG
        self.__sr_SaO2 = sr_SaO2

    def __len__(self):
        return len(self.__y)

    def __getitem__(self, idx:int):
        return self.__X_EEG[idx], self.__X_SaO2[idx], self.__y[idx]
   
    def signal_len_EEG(self):
        return self.__X_EEG.shape[1]
    
    def signal_len_SaO2(self):
        return self.__X_SaO2.shape[1]

    def split_dataset(self):
        """
        Splits the dataset into 10 subsets, ensuring that each subset contains a balanced distribution of classes.
        It modifies the `self.subsets` attribute by appending 10 subsets.

        Returns: none
        """

        labels = np.array([self.__y[i].item() for i in range(len(self))]) #Extract the labels from the dataset
       
        other_indices = np.arange(len(labels)) #Create an array of indices corresponding to the dataset
        for i in range(0,9): #Get 9 stratified splits, leaving the last 10% for the final subset
            other_indices, set_indices = train_test_split(
                other_indices,
                test_size=round(0.1/(1-0.1*i), 2), #Calculate test size to ensure 10% of the remaining data is split off
                stratify=labels[other_indices] #Stratified split to maintain class balance
            )
            self.subsets.append(Subset(self, set_indices))
        self.subsets.append(Subset(self, other_indices))
   
    def get_subsets(self, idxs):
        indices = []
        for idx in idxs:
            subset_indices = list(self.subsets[idx].indices)
            indices.extend(subset_indices)
        return Subset(self.subsets[idx].dataset, indices)

    def data_analysis(self):
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
        """
        Undersamples the majority class in specified subsets to achieve a balanced class distribution. It modifies `self.subsets` in place.

        Parameters:
        - majority_class: The label of the majority class that needs to be undersampled.
        - subsets: A list of indices specifying which subsets to process.
        - prop (float): Proportion of the minimum class count to which the majority class will be undersampled (default is 1, meaning the majority class is reduced to the size of the smallest class).

        Returns: none
        """
        for subset in subsets:
            class_counts = Counter([int(self.subsets[subset][i][2].item()) for i in range(len(self.subsets[subset]))]) #Count the number of samples for each class in the subset
            min_count = min(class_counts.values()) #Determine the minimum number of samples in any class
            indices = []
            class_counter = {cls: 0 for cls in class_counts}

            for idx in range(len(self.subsets[subset])):
                label = int(self.subsets[subset][idx][2].item())
                if label == majority_class:
                    #If the sample is from the majority class and we haven't reached the proportional limit, include it
                    if class_counter[label] < min_count*prop:
                        indices.append(self.subsets[subset].indices[idx])
                        class_counter[label] += 1
                else:
                    #Include all samples from other classes
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
            'X_EEG': self.__X_EEG,
            'X_SaO2': self.__X_SaO2,
            'y': self.__y,
            'sr_EEG': self.__sr_EEG,
            'sr_SaO2': self.__sr_SaO2,
            'archivos': self.archivos,
            'subsets': self.subsets if len(self.subsets)>0 else []
        }
        torch.save(data, file_path)

    def resample_segments_EEG(self, sampling_rate):
        """
        Resamples the signal segments to a uniform length based on the specified sampling rate. It modifies `self.__X` and updates the subsets' datasets in place.

        Parameters:
        - sampling_rate: The desired sampling rate (in Hz) to resample the segments.

        Returns: none
        """
        new_X = []
        segment_length = 30
        num_puntos = segment_length * sampling_rate
        for segment in self.__X_EEG:
            new_segment = resample(segment.numpy(), num_puntos)
            new_X.append(new_segment)
        new_X_np = np.array(new_X)
        self.__X_EEG = torch.tensor(new_X_np)
        for idx, subset in enumerate(self.subsets):
            self.subsets[idx].dataset = self

    def resample_segments_SaO2(self, sampling_rate):
        """
        Resamples the signal segments to a uniform length based on the specified sampling rate. It modifies `self.__X` and updates the subsets' datasets in place.

        Parameters:
        - sampling_rate: The desired sampling rate (in Hz) to resample the segments.

        Returns: none
        """
        new_X = []
        segment_length = 30
        num_puntos = segment_length * sampling_rate
        for segment in self.__X_SaO2:
            new_segment = resample(segment.numpy(), num_puntos)
            new_X.append(new_segment)
        new_X_np = np.array(new_X)
        self.__X_SaO2 = torch.tensor(new_X_np)
        for idx, subset in enumerate(self.subsets):
            self.subsets[idx].dataset = self


    @staticmethod
    def join_datasets(datasets, traintestval=None):
        """
        Merges multiple instances of `ApneaDataset2` into a single dataset, combining their data and subsets.

        Parameters:
        - datasets: A list of `ApneaDataset2` instances to be joined.
        - traintestval (tuple of lists, optional): Contains train, validation, and test subsets for each dataset in `datasets`. It should be a list of tuples where each tuple has three lists (train, val, test subsets). Only required if you want to merge the subsets along with the datasets.

        Returns:
        - A tuple containing:
        1. The merged `ApneaDataset2` instance.
        2. Updated train subset indices.
        3. Updated validation subset indices.
        4. Updated test subset indices.
        """

        if not isinstance(datasets[0], ApneaDataset_SaO2):
            raise TypeError("Expected an instance of ApneaDataset")
        appended_dataset = copy.deepcopy(datasets[0]) #Create a deep copy of the first dataset to serve as the base
        new_subsets = []
        offset = 0
        if traintestval is not None:
            train_subsets, val_subsets, test_subsets = traintestval[0]

        for idx, dataset in enumerate(datasets):
            if not isinstance(dataset, ApneaDataset_SaO2):
                raise TypeError("Expected an instance of ApneaDataset2")
            if idx > 0:  # Skip the first dataset since it's already copied
                #concatenate their data
                appended_dataset.__X_EEG = torch.cat((appended_dataset.__X_EEG, dataset.__X_EEG), dim=0)
                appended_dataset.__X_SaO2 = torch.cat((appended_dataset.__X_SaO2, dataset.__X_SaO2), dim=0)
                appended_dataset.__y = torch.cat((appended_dataset.__y, dataset.__y.clone().detach()), dim=0)
                try:
                    appended_dataset.archivos.append(dataset.archivos)
                except AttributeError:
                    appended_dataset.archivos = [appended_dataset.archivos] + [dataset.archivos]

            if(traintestval != None and idx > 0):
                train_subsets_old, val_subsets_old, test_subsets_old = traintestval[idx]
                train_subsets += [x+len(new_subsets) for x in train_subsets_old]
                val_subsets += [x+len(new_subsets) for x in val_subsets_old]
                test_subsets += [x+len(new_subsets) for x in test_subsets_old]

            for subset in dataset.subsets:
                #Adjust indices for the new subsets and add them to the new_subsets list
                new_indices = [i + offset for i in subset.indices]
                new_subsets.append(Subset(appended_dataset, new_indices))

            offset += len(dataset.__X_EEG)

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
        apnea_dataset = ApneaDataset_SaO2(data['X_EEG'], data['X_SaO2'], data['y'], data['sr_EEG'], data['sr_SaO2'], data['archivos'])
        if len(data['subsets'])>0:
            apnea_dataset.subsets = data['subsets']
        return apnea_dataset

    @staticmethod
    def from_segments_SaO2(segments_EEG, segments_SaO2):
        X_EEG = []
        X_SaO2 = []
        y = []

        if(len(segments_EEG) == len(segments_SaO2)):
            for idx in range(0, len(segments_EEG)):
                X_EEG.append(segments_EEG[idx]['Signal'])
                X_SaO2.append(segments_SaO2[idx]['Signal'])
                if(segments_EEG[idx]['Label'] == segments_SaO2[idx]['Label']):
                    y.append(segments_EEG[idx]['Label'])
        else:
            print("LONGITUDES DISTINTAS")

        # Convertimos las listas en arreglos numpy para que sea más rápido
        X_EEG = np.array(X_EEG)
        X_SaO2 = np.array(X_SaO2)
        y = np.array(y)

        X_EEG = torch.tensor(X_EEG, dtype=torch.float32)#.unsqueeze(1)
        X_SaO2 = torch.tensor(X_SaO2, dtype=torch.float32)#.unsqueeze(1)
        y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

        return X_EEG, X_SaO2, y

    def create_datasets_SaO2(files, path_edf, path_annot, overlap, perc_apnea, filtering = False, filter = ""):
        for file in files:
            all_signals = read_signals_EDF(path_edf + f"\\homepap-lab-full-1600{str(file).zfill(3)}.edf")
            annotations = get_annotations(path_annot + f"\\homepap-lab-full-1600{str(file).zfill(3)}-profusion.xml")

            signal_EEG, time_EEG, sampling_EEG = get_bipolar_signal(all_signals['C3'], all_signals['O1']) 

            if(filtering):
                if(filter == "Butter"):
                    signal_EEG = filter_Butter(signal_EEG, sampling_EEG, plot = False)
                elif(filter == "FIR"):
                    signal_EEG = filter_FIR2(signal_EEG, sampling_EEG, plot = False)
                elif(filter == "Notch_FIR"):
                    signal_EEG = filter_Notch_FIR(signal_EEG, sampling_EEG, plot = False)
                    signal_EEG = filter_FIR2(signal_EEG, sampling_EEG, plot = False)
                elif(filter == "FIR_Notch"):
                    signal_EEG = filter_FIR2(signal_EEG, sampling_EEG, plot = False)
                    signal_EEG = filter_Notch_FIR(signal_EEG, sampling_EEG, plot = False)
                elif(filter == "Notch"):
                    signal_EEG = filter_Notch_FIR(signal_EEG, sampling_EEG, plot = False)

            segments_EEG = get_signal_segments(signal_EEG, time_EEG, sampling_EEG, annotations, period_length=30, overlap=overlap, perc_apnea=perc_apnea, t_descarte = 5*60)

            try:
                SaO2 = all_signals['SaO2']
            except:
                SaO2 = all_signals['SAO2']
            signal_SaO2, time_SaO2, sampling_SaO2 = SaO2['Signal'], SaO2['Time'], SaO2['SamplingRate']

            segments_SaO2 = get_signal_segments(signal_SaO2, time_SaO2, sampling_SaO2, annotations, period_length=30, overlap=overlap, perc_apnea=perc_apnea, t_descarte = 5*60)

            X_EEG, X_SaO2, y = ApneaDataset_SaO2.from_segments_SaO2(segments_EEG, segments_SaO2)
            dataset = ApneaDataset_SaO2(X_EEG, X_SaO2, y, sampling_EEG, sampling_SaO2, file)
            dataset.split_dataset()
            dataset.save_dataset(f".\data\ApneaDetection_HomePAPSignals\datasets\dataset_file_1600{file:03d}_EEG_C3O1_SaO2.pth")


    def Zscore_normalization(self):
        """
        Applies Z-score normalization to the input data (self.__X). 
        The normalization scales the features of the data such that they have a mean of 0 and a standard deviation of 1.
        """
        for i in range(self.__X.shape[1]):
            scaler = StandardScaler()
            flat_X = []
            for segment in self.__X:
                flat_X.append(segment[i].numpy())
            scaler.fit(flat_X)
            scaled_X = scaler.transform(flat_X)
            for idx in range(len(self.__X)):
                self.__X[idx][i] = torch.tensor(scaled_X[idx])
