from Imports import *
from Models import Model, DualInputModel

class Trainer:
    """Class to train the model with trainset and validate training with valset"""

    def __init__(self, 
                 model:Model, 
                 trainset:Subset, 
                 valset:Subset, 
                 n_epochs:int = 100, 
                 batch_size:int = 32, 
                 loss_fn:str = 'BCE', 
                 optimizer:str = 'SGD', 
                 lr:float = 0.01, 
                 momentum:float = 0, 
                 text:str = '', 
                 device = "cpu"):
        """
        Initializes the Trainer object.

        Args:
            - model (Model): model to test.
            - trainset (Subset): data to train.
            - valset (Subset): data to validate.
            - n_epochs (int): number of epochs to train the model.
            - batch_size (int): number of data used in one iteration.
            - loss_fn (str): string to specify desired loss function. NOTE: in this version, only 'BCE' and 'CrossEntropyLoss' loss functions are available.
            - optimizer (str): string to specify desired optimizer. NOTE: in this version, only 'SGD' and 'Adam' optimizers are available.
            - lr (float): learning rate for the optimizer.
            - momentum (float): for the optimizer. NOTE: not used if optimizer is Adam
            - txt (str): previous data statistics to save in the .txt file.
        
        Returns: none.
        """

        self.device = device

        self.__model__ = model.to(self.device)

        self.train_loader = DataLoader(trainset, shuffle=True, batch_size=batch_size)
        self.val_loader = DataLoader(valset, shuffle=False, batch_size=batch_size)

        if loss_fn == 'BCE':
            self.loss_fn = nn.BCELoss()  # Binary Cross Entropy
        elif loss_fn == 'CE':
            self.loss_fn = nn.CrossEntropyLoss()  # Cross Entropy
        else:
            raise Exception("In this version, only 'BCE' or 'CE' loss functions are available")
        if optimizer == 'SGD':
            self.optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)  # Stochastic Gradient Descent
        elif optimizer == 'Adam':
            self.optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999)) # Adam
        else:
            raise Exception("In this version, only 'SGD' or 'Adam' optimizers are available")
        
        self.batch_size = batch_size
        self.lr = lr
        self.n_epochs = n_epochs
        self.train_accuracies = []
        self.train_losses = []
        self.val_accuracies = []
        self.val_losses = []
        self.f1score = []
        self.momentum = momentum
        self.text = text


    def train_one_epoch(self):
        """
        Trains the model for one epoch and returns the average loss and accuracy.

        Args: none.

        Returns:
        - avg_loss (float): average loss over the epoch.
        - train_acc (float): accuracy of the model on the training data.
        """

        #Set model in training model and initialize variables:
        self.__model__.train()
        running_loss = 0.0
        all_y_true = []
        all_y_pred = []

        #Iterate over each batch in the train_loader:
        for X_batch, y_batch in self.train_loader:
            
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            #Get the model's predictions for the batch and calculate the loss:
            y_pred = self.__model__(X_batch)
            loss = self.loss_fn(y_pred, y_batch)

            #Reset and calculate gradients via backpropagation and update the model's parameters:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            #Accumulate the loss and store the true labels and predictions for accuracy calculation later:
            running_loss += loss.item()
            all_y_true.append(y_batch)
            all_y_pred.append(y_pred)

        #Calculate the average loss and accuracy for the epoch:
        avg_loss = running_loss / len(self.train_loader)
        y_true = torch.cat(all_y_true).detach().cpu().numpy().tolist()
        y_pred = torch.cat(all_y_pred).round().detach().cpu().numpy().tolist()
        train_acc = accuracy_score(y_true, y_pred)
        return avg_loss, train_acc

    def val_one_epoch(self):
        """
        Evaluates the model for one epoch and returns the average validation loss, accuracy, and F1 score.

        Args: none.

        Returns:
        - avg_val_loss (float): average loss over the validation epoch.
        - val_acc (float): accuracy of the model on the validation data.
        - f1 (float): F1 score of the model on the validation data.
        """

        #Set model in evaluation model and initialize variables:
        self.__model__.eval()
        val_running_loss = 0.0
        all_y_true = []
        all_y_pred = []

        #Without gradient calculation:
        with torch.no_grad():
            #Iterate over each batch in the val_loader:
            for X_val, y_val in self.val_loader:
                X_val, y_val = X_val.to(self.device), y_val.to(self.device)
                #Get the model's predictions for the batch and calculate the loss:
                y_pred = self.__model__(X_val)
                val_loss = self.loss_fn(y_pred, y_val)
                #Accumulate the loss and store the true labels and predictions for accuracy calculation later:
                val_running_loss += val_loss.item()
                all_y_true.append(y_val)
                all_y_pred.append(y_pred)
        #Calculate the average validation loss, accuracy and F1 score for the epoch:
        avg_val_loss = val_running_loss / len(self.val_loader)
        y_true = torch.cat(all_y_true)
        y_pred = torch.cat(all_y_pred).round()
        val_acc = accuracy_score(y_true.cpu(), y_pred.cpu())
        f1 = f1_score(y_true.cpu(), y_pred.cpu())

        return avg_val_loss, val_acc, f1

    def train(self, models_path, verbose:bool = True, plot:bool = True, save_model = True, save_best_model = True):
        """
        Trains the model for a specified number of epochs, optionally printing progress.

        Args:
            - verbose (bool): If True, prints training progress after each epoch. If False, it will not print to console but save in the ```models\nombre\nombre_training.txt``` file
            - plot (bool): If True, plots training and validation accuracies and losses after training. If False, the figures will not be displayed but saved in the ```models\nombre\nombre_''.png``` files
        
        Returns:
            - model (Model): the trained model.
        """
        #Track the start time of the entire training process:
        start_time_training = datetime.now()
        self.text += f'\n\nBatch size: {self.batch_size}\n\nLoss function: {self.loss_fn}\n\nOptimizer: {self.optimizer}'
        self.text += '\n\nStart of training'
        min_loss = 1000000
        epoch_min_loss = 0
        # For each epoch:
        for epoch in range(self.n_epochs):
            start_time_epoch = datetime.now()

            # Training: train the model for one epoch, recording the average training loss and accuracy.
            avg_train_loss, train_acc = self.train_one_epoch()
            self.train_losses.append(avg_train_loss)
            self.train_accuracies.append(train_acc)

            # Validation: validate the model for one epoch, recording the average validation loss, accuracy, and F1 score.
            avg_val_loss, val_acc, f1 = self.val_one_epoch()
            self.val_losses.append(avg_val_loss)
            self.val_accuracies.append(val_acc)
            self.f1score.append(f1)
            #Log and optionally print the results of each epoch:
            end_of_epoch = (f"End of epoch {epoch + 1} - Accuracy = {val_acc * 100:.2f}% - F1 = {f1 * 100:.2f}% "
                            f"- Train Loss = {avg_train_loss*100:.2f}% - Val Loss = {avg_val_loss*100:.2f}% - "
                            f"{(datetime.now()-start_time_epoch).total_seconds():.2f} seconds")
            if(verbose):
                print(end_of_epoch)
            self.text += '\n\t' + end_of_epoch
            if(avg_val_loss < min_loss and save_best_model):
                min_loss = avg_val_loss
                epoch_min_loss = epoch + 1
                self.__model__.save_model(models_path, extension = '_best.pth')

        #Track the end time of the entire training process:
        end_of_training = (f"End of training - {self.n_epochs} epochs - "
                           f"{(datetime.now()-start_time_training).total_seconds():.2f} seconds")
        if(save_best_model):
            end_of_training += f"\nBest model - Epoch {epoch_min_loss} - Val Loss = {min_loss*100:.2f}%"
        if(verbose):
            print(end_of_training)
        self.text += '\n' + end_of_training
        #Optionally plot the accuracies and losses over epochs:
        self.plot_accuracies_losses(models_path, plot)
        #Write the training logs to a text file:
        self.write_txt(models_path)
        if(save_model):
            self.__model__.save_model(models_path)
        return self.__model__

    def plot_accuracies_losses(self, models_path, plot:bool):
        """
        Plots accuracy, F1 score and loss vs epochs. 
        
        Args:
            - plot (bool): If False the figures will not be displayed but will be saved in 'models\\nombre\\nombre_acc_loss.png'.
        
        Returns: none
        """

        plt.figure(figsize=(12, 5))

        #Accuracy
        plt.subplot(1, 3, 1)
        plt.plot(range(1, self.n_epochs + 1), self.train_accuracies, marker='o', color='blue', label='Train')
        plt.plot(range(1, self.n_epochs + 1), self.val_accuracies, marker='o', color='green', label='Val')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Epoch vs Accuracy')
        #F1score
        plt.subplot(1, 3, 2)
        plt.plot(range(1, self.n_epochs + 1), self.f1score, marker='o', color='brown', label='Val')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.title('Epoch vs F1 Score')
        #Loss
        plt.subplot(1, 3, 3)
        plt.plot(range(1, self.n_epochs + 1), self.val_losses, marker='o', color='orange', label='Val')
        plt.plot(range(1, self.n_epochs + 1), self.train_losses, marker='o', color='red', label='Train')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Epoch vs Loss')

        plt.tight_layout()

        if os.path.exists(models_path):
            PATH = models_path + f'/{self.__model__.get_name()}_acc_loss.png'
            plt.savefig(PATH)

        if(plot):
            plt.show()  
        else:
            plt.close()

    def write_txt(self, models_path):
        """
        Write .txt file with training information:
        - Data distribution
        - Model architecture
        - Batch size
        - Loss function
        - Optimizer
        - Accuracy, F1score, Loss per epoch
        
        Args: none.
        
        Returns: none.
        """           
        if os.path.exists(models_path):
            PATH = models_path + f'/{self.__model__.get_name()}_training.txt'
            f = open(PATH, "w")
            f.write(self.text)
            f.close()





class Trainer_SaO2:
    """Class to train the model with trainset and validate training with valset"""

    def __init__(self, 
                 model:DualInputModel, 
                 trainset:Subset, 
                 valset:Subset, 
                 n_epochs:int = 100, 
                 batch_size:int = 32, 
                 loss_fn:str = 'BCE', 
                 optimizer:str = 'SGD', 
                 lr:float = 0.01, 
                 momentum:float = 0, 
                 text:str = '', 
                 device = "cpu"):
        """
        Initializes the Trainer object.

        Args:
            - model (Model): model to test.
            - trainset (Subset): data to train.
            - valset (Subset): data to validate.
            - n_epochs (int): number of epochs to train the model.
            - batch_size (int): number of data used in one iteration.
            - loss_fn (str): string to specify desired loss function. NOTE: in this version, only 'BCE' and 'CrossEntropyLoss' loss functions are available.
            - optimizer (str): string to specify desired optimizer. NOTE: in this version, only 'SGD' and 'Adam' optimizers are available.
            - lr (float): learning rate for the optimizer.
            - momentum (float): for the optimizer. NOTE: not used if optimizer is Adam
            - txt (str): previous data statistics to save in the .txt file.
        
        Returns: none.
        """

        self.device = device

        self.__model__ = model.to(self.device)

        self.train_loader = DataLoader(trainset, shuffle=True, batch_size=batch_size)
        self.val_loader = DataLoader(valset, shuffle=False, batch_size=batch_size)

        if loss_fn == 'BCE':
            self.loss_fn = nn.BCELoss()  # Binary Cross Entropy
        elif loss_fn == 'CE':
            self.loss_fn = nn.CrossEntropyLoss()  # Cross Entropy
        else:
            raise Exception("In this version, only 'BCE' or 'CE' loss functions are available")
        if optimizer == 'SGD':
            self.optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)  # Stochastic Gradient Descent
        elif optimizer == 'Adam':
            self.optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999)) # Adam
        else:
            raise Exception("In this version, only 'SGD' or 'Adam' optimizers are available")
        
        self.batch_size = batch_size
        self.lr = lr
        self.n_epochs = n_epochs
        self.train_accuracies = []
        self.train_losses = []
        self.val_accuracies = []
        self.val_losses = []
        self.f1score = []
        self.momentum = momentum
        self.text = text


    def train_one_epoch(self):
        """
        Trains the model for one epoch and returns the average loss and accuracy.

        Args: none.

        Returns:
        - avg_loss (float): average loss over the epoch.
        - train_acc (float): accuracy of the model on the training data.
        """

        #Set model in training model and initialize variables:
        self.__model__.train()
        running_loss = 0.0
        all_y_true = []
        all_y_pred = []

        #Iterate over each batch in the train_loader:
        for X_EEG_batch, X_SaO2_batch, y_batch in self.train_loader:
            
            X_EEG_batch, X_SaO2_batch, y_batch = X_EEG_batch.to(self.device), X_SaO2_batch.to(self.device), y_batch.to(self.device)
            #Get the model's predictions for the batch and calculate the loss:
            y_pred = self.__model__(X_EEG_batch, X_SaO2_batch)
            loss = self.loss_fn(y_pred, y_batch)

            #Reset and calculate gradients via backpropagation and update the model's parameters:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            #Accumulate the loss and store the true labels and predictions for accuracy calculation later:
            running_loss += loss.item()
            all_y_true.append(y_batch)
            all_y_pred.append(y_pred)

        #Calculate the average loss and accuracy for the epoch:
        avg_loss = running_loss / len(self.train_loader)
        y_true = torch.cat(all_y_true).detach().cpu().numpy().tolist()
        y_pred = torch.cat(all_y_pred).round().detach().cpu().numpy().tolist()
        train_acc = accuracy_score(y_true, y_pred)
        return avg_loss, train_acc

    def val_one_epoch(self):
        """
        Evaluates the model for one epoch and returns the average validation loss, accuracy, and F1 score.

        Args: none.

        Returns:
        - avg_val_loss (float): average loss over the validation epoch.
        - val_acc (float): accuracy of the model on the validation data.
        - f1 (float): F1 score of the model on the validation data.
        """

        #Set model in evaluation model and initialize variables:
        self.__model__.eval()
        val_running_loss = 0.0
        all_y_true = []
        all_y_pred = []

        #Without gradient calculation:
        with torch.no_grad():
            #Iterate over each batch in the val_loader:
            for X_EEG_val, X_SaO2_val, y_val in self.val_loader:
                X_EEG_val, X_SaO2_val, y_val = X_EEG_val.to(self.device), X_SaO2_val.to(self.device), y_val.to(self.device)
                #Get the model's predictions for the batch and calculate the loss:
                y_pred = self.__model__(X_EEG_val, X_SaO2_val)
                val_loss = self.loss_fn(y_pred, y_val)
                #Accumulate the loss and store the true labels and predictions for accuracy calculation later:
                val_running_loss += val_loss.item()
                all_y_true.append(y_val)
                all_y_pred.append(y_pred)
        #Calculate the average validation loss, accuracy and F1 score for the epoch:
        avg_val_loss = val_running_loss / len(self.val_loader)
        y_true = torch.cat(all_y_true)
        y_pred = torch.cat(all_y_pred).round()
        val_acc = accuracy_score(y_true.cpu(), y_pred.cpu())
        f1 = f1_score(y_true.cpu(), y_pred.cpu())

        return avg_val_loss, val_acc, f1

    def train(self, models_path, verbose:bool = True, plot:bool = True, save_model = True, save_best_model = True):
        """
        Trains the model for a specified number of epochs, optionally printing progress.

        Args:
            - verbose (bool): If True, prints training progress after each epoch. If False, it will not print to console but save in the ```models\nombre\nombre_training.txt``` file
            - plot (bool): If True, plots training and validation accuracies and losses after training. If False, the figures will not be displayed but saved in the ```models\nombre\nombre_''.png``` files
        
        Returns:
            - model (Model): the trained model.
        """
        #Track the start time of the entire training process:
        start_time_training = datetime.now()
        self.text += f'\n\nBatch size: {self.batch_size}\n\nLoss function: {self.loss_fn}\n\nOptimizer: {self.optimizer}'
        self.text += '\n\nStart of training'
        min_loss = 1000000
        epoch_min_loss = 0
        # For each epoch:
        for epoch in range(self.n_epochs):
            start_time_epoch = datetime.now()

            # Training: train the model for one epoch, recording the average training loss and accuracy.
            avg_train_loss, train_acc = self.train_one_epoch()
            self.train_losses.append(avg_train_loss)
            self.train_accuracies.append(train_acc)

            # Validation: validate the model for one epoch, recording the average validation loss, accuracy, and F1 score.
            avg_val_loss, val_acc, f1 = self.val_one_epoch()
            self.val_losses.append(avg_val_loss)
            self.val_accuracies.append(val_acc)
            self.f1score.append(f1)
            #Log and optionally print the results of each epoch:
            end_of_epoch = (f"End of epoch {epoch + 1} - Accuracy = {val_acc * 100:.2f}% - F1 = {f1 * 100:.2f}% "
                            f"- Train Loss = {avg_train_loss*100:.2f}% - Val Loss = {avg_val_loss*100:.2f}% - "
                            f"{(datetime.now()-start_time_epoch).total_seconds():.2f} seconds")
            if(verbose):
                print(end_of_epoch)
            self.text += '\n\t' + end_of_epoch
            if(avg_val_loss < min_loss and save_best_model):
                min_loss = avg_val_loss
                epoch_min_loss = epoch + 1
                self.__model__.save_model(models_path, extension = '_best.pth')

        #Track the end time of the entire training process:
        end_of_training = (f"End of training - {self.n_epochs} epochs - "
                           f"{(datetime.now()-start_time_training).total_seconds():.2f} seconds")
        if(save_best_model):
            end_of_training += f"\nBest model - Epoch {epoch_min_loss} - Val Loss = {min_loss*100:.2f}%"
        if(verbose):
            print(end_of_training)
        self.text += '\n' + end_of_training
        #Optionally plot the accuracies and losses over epochs:
        self.plot_accuracies_losses(models_path, plot)
        #Write the training logs to a text file:
        self.write_txt(models_path)
        if(save_model):
            self.__model__.save_model(models_path)
        return self.__model__

    def plot_accuracies_losses(self, models_path, plot:bool):
        """
        Plots accuracy, F1 score and loss vs epochs. 
        
        Args:
            - plot (bool): If False the figures will not be displayed but will be saved in 'models\\nombre\\nombre_acc_loss.png'.
        
        Returns: none
        """

        plt.figure(figsize=(12, 5))

        #Accuracy
        plt.subplot(1, 3, 1)
        plt.plot(range(1, self.n_epochs + 1), self.train_accuracies, marker='o', color='blue', label='Train')
        plt.plot(range(1, self.n_epochs + 1), self.val_accuracies, marker='o', color='green', label='Val')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Epoch vs Accuracy')
        #F1score
        plt.subplot(1, 3, 2)
        plt.plot(range(1, self.n_epochs + 1), self.f1score, marker='o', color='brown', label='Val')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.title('Epoch vs F1 Score')
        #Loss
        plt.subplot(1, 3, 3)
        plt.plot(range(1, self.n_epochs + 1), self.val_losses, marker='o', color='orange', label='Val')
        plt.plot(range(1, self.n_epochs + 1), self.train_losses, marker='o', color='red', label='Train')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Epoch vs Loss')

        plt.tight_layout()

        if os.path.exists(models_path):
            PATH = models_path + f'/{self.__model__.get_name()}_acc_loss.png'
            plt.savefig(PATH)

        if(plot):
            plt.show()  
        else:
            plt.close()

    def write_txt(self, models_path):
        """
        Write .txt file with training information:
        - Data distribution
        - Model architecture
        - Batch size
        - Loss function
        - Optimizer
        - Accuracy, F1score, Loss per epoch
        
        Args: none.
        
        Returns: none.
        """           
        if os.path.exists(models_path):
            PATH = models_path + f'/{self.__model__.get_name()}_training.txt'
            f = open(PATH, "w")
            f.write(self.text)
            f.close()

