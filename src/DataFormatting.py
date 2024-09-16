from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.signal import resample
import copy
from Imports import *
from RealSignals import *
from Annotations import get_annotations

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
        for segment in self.__X:
            new_segment = [resample(segment[0].numpy(), num_puntos)]
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
    def from_segments(segments):
        X = []
        y = []

        for segment in segments:
            X.append(segment['Signal'])
            y.append(segment['Label'])

        # Convertimos las listas en arreglos numpy
        X = np.vstack(X)
        y = np.array(y)

        X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

        return X,y

    @staticmethod
    def create_datasets(files, path_edf, path_annot):
        """
        Creates datasets from EDF files and annotations by processing each file, generating segments, and saving them.

        Parameters:
        - files: A list of file identifiers to process.
        - path_edf: The path to the directory containing EDF files.
        - path_annot: The path to the directory containing annotation files.

        Returns: none
        """
        for file in files:
            all_signals = read_signals_EDF(path_edf + f"\\homepap-lab-full-1600{str(file).zfill(3)}.edf")
            annotations = get_annotations(path_annot + f"\\homepap-lab-full-1600{str(file).zfill(3)}-profusion.xml")
            bipolar_signal, time, sampling = get_bipolar_signal(all_signals['C3'], all_signals['O1'])
            segments = get_signal_segments_strict(bipolar_signal, time, sampling, annotations)

            X,y = ApneaDataset.from_segments(segments)
            dataset = ApneaDataset(X, y, sampling, file)
            dataset.split_dataset()
            dataset.save_dataset(f".\data\ApneaDetection_HomePAPSignals\datasets\dataset2_archivo_1600{file:03d}.pth")


    def Zscore_normalization(self):
        """
        Applies Z-score normalization to the input data (self.__X). 
        The normalization scales the features of the data such that they have a mean of 0 and a standard deviation of 1.
        """
        scaler = StandardScaler()
        original_shape = self.__X.shape
        flattened_X = self.__X.reshape(-1, original_shape[-1])
        scaler.fit(flattened_X)
        scaled_X = scaler.transform(flattened_X)
        self.__X = torch.tensor(scaled_X, dtype=torch.float32).view(original_shape)
