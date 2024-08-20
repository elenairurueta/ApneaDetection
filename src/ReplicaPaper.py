from Imports import *
from DataFormatting import ApneaDataset2

class Model3(nn.Module):
    """Neural network model class"""
    def __init__(self, input_size, nombre:str, kernel_length_1 = 35, n_filters_1 = 8, kernel_length_2 = 175, n_filters_2 = 128, kernel_length_3 = 175, n_filters_3 = 16, maxpool = 7, dense_nodes = 64, conv_dropout = 0.1, dense_dropout = 0):
        """
        Initializes the neural network model.

        Args:
            input_size (int): size of the input data.
            nombre (str): name of the model.
        """

        super().__init__()
        self.nombre = nombre

        def conv_output_size(input_size, kernel_size, padding, stride, dilation = 1):
            return int(np.floor((input_size + 2*padding - dilation*(kernel_size - 1) - 1)/stride + 1))
        
        def maxpool_output_size(input_size, kernel_size, stride, padding = 0, dilation = 1):
            return int(np.floor((input_size + 2*padding - dilation*(kernel_size - 1) - 1)/stride + 1))

        size = input_size
        size = conv_output_size(size, kernel_length_1, padding=0, stride=1)
        size = maxpool_output_size(size, kernel_size=maxpool, stride=1)
        size = conv_output_size(size, kernel_length_2, padding=0, stride=1)
        size = maxpool_output_size(size, kernel_size=maxpool, stride=1)
        size = conv_output_size(size, kernel_length_3, padding=0, stride=1)
        size = maxpool_output_size(size, kernel_size=maxpool, stride=1)

        #Convolutional layers:
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, n_filters_1, kernel_size=kernel_length_1, stride=1, padding='same'), #(input_size - (35-1) - 1)/1 + 1 = input_size - 34
            nn.BatchNorm1d(n_filters_1),
            nn.ELU(),
            nn.Dropout(conv_dropout),
            nn.MaxPool1d(kernel_size=maxpool, stride=maxpool), #(input_size - 34) - 7 + 1 = input_size - 40

            nn.Conv1d(n_filters_1, n_filters_2, kernel_size=kernel_length_2, stride=1, padding='same'), #(input_size - 40) - 175 + 1 = input_size - 214
            nn.BatchNorm1d(n_filters_2),
            nn.ELU(),
            nn.Dropout(conv_dropout),
            nn.MaxPool1d(kernel_size=maxpool, stride=maxpool), #(input_size - 214) - 7 + 1 = input_size - 220

            nn.Conv1d(n_filters_2, n_filters_3, kernel_size=kernel_length_3, stride=1, padding='same'), #(input_size - 220) - 175 + 1 = input_size - 394
            nn.BatchNorm1d(n_filters_3),
            nn.ELU(),
            nn.Dropout(conv_dropout),
            nn.MaxPool1d(kernel_size=maxpool, stride=maxpool) #(input_size - 394) - 7 + 1 = input_size - 400
        )

        self.conv_output_size = self._get_conv_output_size(input_size)
        self.fc_layers = nn.Sequential(
            nn.Linear(self.conv_output_size, dense_nodes),
            nn.ELU(),
            nn.Dropout(dense_dropout),
            nn.Linear(dense_nodes, 2),
            nn.Softmax(dim = 1)
        )
        


    def _get_conv_output_size(self, input_size):
        x = torch.randn(1, 1, input_size)
        x = self.conv_layers(x)
        return x.numel() // x.size(0)
    def forward(self, x):
        """
        Defines the forward pass of the neural network model.
        """
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x  
    def get_nombre(self):
        """
        Args: none.

        Returns:
            - str: the name of the model.
        """
        return self.nombre

    def get_architecture(self):
        """
        Args: none.

        Returns:
            - str: the architecture of the model.
        """
        return '\n\n' + str(self)

    def save_model(self, models_path, extension:str = '.pth'):
        """
        Saves the parameters of the model to a file.

        Args:
            - extension (str, optional): extension of the file ('.pt' or '.pth'). Defaults to '.pth'.

        Returns: none
        """
        if os.path.exists(models_path):
            if not os.path.exists(models_path + f'/{self.nombre}'):
                os.makedirs(models_path + f'/{self.nombre}')
            PATH = models_path + f'/{self.nombre}/{self.nombre + extension}'             
        torch.save(self.state_dict(), PATH)

    @staticmethod
    def load_model(models_path, nombre, input_size, extension:str = '.pth', best = False):
        """
        Loads a pre-trained model from a file.

        Args:
            - nombre (str): name of the model.
            - input_size (int): size of the input data.
            - extension (str, optional): extension of the file ('.pt' or '.pth'). Defaults to '.pth'.
            
        Returns:
            - Model: the loaded model.
        """
        model = Model3(input_size, nombre)
        if(best):
            PATH = models_path + f'/{nombre}/{nombre}_best{extension}' 
            model.load_state_dict(torch.load(PATH))
        else:
            PATH = models_path + f'/{nombre}/{nombre + extension}'
            model.load_state_dict(torch.load(PATH))
        model.eval()
        return model

class Trainer:
    """Class to train the model with trainset and validate training with valset"""

    def __init__(self, 
                 model:Model3, 
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
            - loss_fn (str): string to specify desired loss function. NOTE: in this first version, only 'BCE' loss function is available.
            - optimizer (str): string to specify desired optimizer. NOTE: in this first version, only 'SGD' optimizer is available.
            - lr (float): learning rate for the optimizer.
            - momentum (float): for the optimizer.
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
            raise Exception("In this first version, only 'BCE' or 'CE' loss function is available")
        if optimizer == 'SGD':
            self.optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)  # Stochastic Gradient Descent
        elif optimizer == 'Adam':
            self.optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999)) # Adam
        else:
            raise Exception("In this first version, only 'SGD' or 'Adam' optimizer is available")
        
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
            y_batch = y_batch.view(-1).long() 
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
        y_pred = torch.cat(all_y_pred).round().detach().cpu()
        y_pred = torch.argmax(y_pred, dim = 1).numpy().tolist()
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
                y_val = y_val.view(-1).long()
                val_loss = self.loss_fn(y_pred, y_val)
                #Accumulate the loss and store the true labels and predictions for accuracy calculation later:
                val_running_loss += val_loss.item()
                all_y_true.append(y_val)
                all_y_pred.append(y_pred)
        #Calculate the average validation loss, accuracy and F1 score for the epoch:
        avg_val_loss = val_running_loss / len(self.val_loader)
        y_true = torch.cat(all_y_true).cpu()
        y_pred = torch.cat(all_y_pred).round().cpu()
        y_pred = torch.argmax(y_pred, dim = 1).numpy().tolist()
        # val_acc = accuracy_score(y_true, y_pred)
        # f1 = f1_score(y_true, y_pred)
        val_acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

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
                self.__model__.save_model(models_path, extension='_best.pth')

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
            if not os.path.exists(models_path + f'/{self.__model__.get_nombre()}'): 
                os.makedirs(models_path + f'/{self.__model__.get_nombre()}') 
            PATH = models_path + f'/{self.__model__.get_nombre()}/{self.__model__.get_nombre()}_acc_loss.png'
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
            if not os.path.exists(models_path + f'/{self.__model__.get_nombre()}'): 
                os.makedirs(models_path + f'/{self.__model__.get_nombre()}') 
            PATH = models_path + f'/{self.__model__.get_nombre()}/{self.__model__.get_nombre()}_training.txt'
            f = open(PATH, "w")
            f.write(self.text)
            f.close()

class Tester:
    """Class to test the model with testset"""
    def __init__(self, model:Model3, testset:Subset, batch_size:int=32, device = "cpu", best_final = ""):
        """
        Initializes the Tester object.

        Args:
            - model (Model): model to test.
            - testset (Subset): data to test.
            - batch_size (int): number of data used in one iteration.

        Returns: none.
        """
        self.device = device
        self.__model__ = model.to(self.device)
        self.__test_loader__ = DataLoader(testset, shuffle=False, batch_size=batch_size)
        self.__cm__ = []
        self.__metrics__ = ""
        self.__all_labels__ = []
        self.__all_preds__ = []
        self.__wrong_predictions__ = []
        self.__best_final__ = best_final
        
    def test(self):
        """
        Tests the model on the test dataset, storing predictions and labels.

        Args: none.

        Returns: none.
        """

        #Iterate over each batch in the test_loader:
        for X_test, y_test in self.__test_loader__:
            X_test, y_test = X_test.to(self.device), y_test.to(self.device)
            #Get the model's predictions for the batch:
            y_test_pred = self.__model__(X_test)
            y_test = y_test.view(-1).long()
            y_test_pred = y_test_pred.round()
            y_test_pred = torch.argmax(y_test_pred.detach().cpu(), dim = 1).numpy().tolist()
            self.__all_preds__.extend(y_test_pred)
            self.__all_labels__.extend(y_test.detach().cpu().numpy().tolist())
            # Save instances where the predictions were incorrect:
            self.__save_wrong_predictions__(X_test.cpu(), y_test.cpu(), y_test_pred)

    def __save_wrong_predictions__(self, X_test:Tensor, y_test:Tensor, y_test_pred:Tensor):
        """
        Saves instances where the model's predictions were incorrect.

        Args:
        - X_test (Tensor): The input test data.
        - y_test (Tensor): The true labels of the test data.
        - y_test_pred (Tensor): The model's predictions for the test data.

        Returns: none.
        """

        # Iterate over each instance in the test data:
        for idx, signal in enumerate(X_test):
            #If the prediction does not match the true label, append the instance to the list of wrong predictions
            if y_test[idx] != y_test_pred[idx]:
                self.__wrong_predictions__.append([signal[0], int(y_test_pred[idx]), int(y_test[idx])])

    def __plot_confusion_matrix__(self, plot:bool):
        """
        Plots the confusion matrix and saves plot as .png.

        Args:
        - plot (bool): If True, shows the plot of the confusion matrix. If False the figure will not be displayed but will be saved in the 'models\nombre\nombre_cm.png' file.
        
        Returns: none.
        """

        #Create a ConfusionMatrixDisplay object with the confusion matrix and class labels:
        cm_display = ConfusionMatrixDisplay(confusion_matrix=self.__cm__, display_labels=['without apnea', 'with apnea'])
        cm_display.plot(cmap='Blues')
        plt.title("Confusion Matrix")

        if os.path.exists(f'/home/elena/Desktop/models'):
            if not os.path.exists(f'/home/elena/Desktop/models/{self.__model__.get_nombre()}'): 
                os.makedirs(f'/home/elena/Desktop/models/{self.__model__.get_nombre()}') 
            PATH = f'/home/elena/Desktop/models/{self.__model__.get_nombre()}/{self.__model__.get_nombre()}_cm_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.png'
            plt.savefig(PATH)

        if(plot):
            plt.show()
        else:
            plt.close()

    def __plot_roc_curve__(self, models_path, plot:bool):
        """
        Plots Receiver Operating Characteristic curve and saves plot as .png.

        Args:
        - plot (bool): If True, shows the plot of the ROC curve. If False the figure will not be displayed but will be saved in the 'models\nombre\nombre_roc.png' file.
        
        Returns: none.
        """

        #Compute the False Positive Rate, True Positive Rate, and thresholds for the ROC curve:
        fpr, tpr, thresholds = roc_curve(self.__all_labels__, self.__all_preds__)
        plt.figure(figsize=(13, 6))
        lw = 2
        #Compute the Area Under the Curve (AUC) for the ROC curve:
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, color='darkorange', lw=lw, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        
        if os.path.exists(models_path):
            if not os.path.exists(models_path + f'/{self.__model__.get_nombre()}'):
                os.makedirs(models_path + f'/{self.__model__.get_nombre()}')
            if(self.__best_final__ == ""):
                PATH = models_path + f'/{self.__model__.get_nombre()}/{self.__model__.get_nombre()}_roc_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.png'
            else:
                PATH = models_path + f'/{self.__model__.get_nombre()}/{self.__model__.get_nombre()}_roc_{self.__best_final__}.png'
            plt.savefig(PATH)
        else:
            print('PATH NOT FOUND: ' + models_path)

        if(plot):
            plt.show()
        else:
            plt.close()

    def __plot_metrics_confusion_matrix__(self, models_path, plot:bool):
        """
        Plots the confusion matrix with normalized values and additional metrics, and saves the plot as a .png file.
        
        Args:
            - plot (bool): If True, shows the confusion matrix with metrics. If False the figure will not be displayed but will be saved in the 'models\nombre\nombre_cm_metrics.png' file.
        
        Returns: none.
        """

        #Normalize the confusion matrix to percentage values:
        cm_normalized = self.__cm__.astype('float') / self.__cm__.sum(axis=1)[:, np.newaxis] * 100
        fig, ax = plt.subplots(figsize=(13, 6))
        #Create a ConfusionMatrixDisplay object with the normalized confusion matrix and class labels:
        cm_display = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=['without apnea', 'with apnea'])
        cm_display.plot(cmap='Blues', ax=ax)
        for text in ax.texts:
            text.set_visible(False)
        for i in range(cm_normalized.shape[0]):
            for j in range(cm_normalized.shape[1]):
                text = ax.text(j, i, f'{cm_normalized[i, j]:.2f}%\n{self.__cm__[i, j]:.2f}', ha='center', va='center', color='black')
        ax.set_title("Confusion Matrix")

        #Generate additional metric text to display below the confusion matrix plot:
        metric_text = (f"Test data count: {len(self.__test_loader__.dataset)}\n") + ("\n".join([f"{k}: {float(v)*100:.2f}%" for k, v in list(self.__metrics__.items())[:-2]])) + "\n" + ("\n".join([f"{k}: {float(v):.4f}" for k, v in list(self.__metrics__.items())[-2:]]))
        plt.gcf().text(0.1, 0.1, metric_text, ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
        
        if os.path.exists(models_path):
            if not os.path.exists(models_path + f'/{self.__model__.get_nombre()}'): 
                os.makedirs(models_path + f'/{self.__model__.get_nombre()}') 
            if(self.__best_final__ == ""):
                PATH = models_path + f'/{self.__model__.get_nombre()}/{self.__model__.get_nombre()}_cm_metrics_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.png'
            else:
                PATH = models_path + f'/{self.__model__.get_nombre()}/{self.__model__.get_nombre()}_cm_metrics_{self.__best_final__}.png'
            plt.savefig(PATH)
        else:
            print('PATH NOT FOUND: ' + models_path)

        if(plot):
            plt.show()
        else:
            plt.close()

    def __calculate_metrics__(self):
        """
        Calculates various evaluation metrics based on the predictions and true labels.
        
        Args: none.
        
        Returns: none.
        """

        metrics = {
            "Accuracy": accuracy_score(self.__all_labels__, self.__all_preds__),
            "Precision": precision_score(self.__all_labels__, self.__all_preds__),
            "Sensitivity": recall_score(self.__all_labels__, self.__all_preds__),
            "Specificity": recall_score(self.__all_labels__, self.__all_preds__, pos_label=0),
            "F1": f1_score(self.__all_labels__, self.__all_preds__),
            "MCC": matthews_corrcoef(self.__all_labels__, self.__all_preds__)
        }

        self.__metrics__ = {k: f"{v:.2f}" for k, v in metrics.items()}

    def evaluate(self, models_path, plot:bool = True):
        '''
        Tests the model, computes metrics and plots evaluation results.

        Args:
            - plot (bool): If True, shows the plots of evaluation results. If False, the figures will not be displayed but will be saved.
        
        Returns: none.
        '''
        self.test()
        self.__cm__ = confusion_matrix(self.__all_labels__, self.__all_preds__)
        self.__calculate_metrics__()
        self.__plot_roc_curve__(models_path, plot)
        self.__plot_metrics_confusion_matrix__(models_path, plot)

        return self.__cm__, self.__metrics__


def init_weights(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

device = torch.device("cpu")
models_path = './models'
nombre0 = f'modelo_replica_paper_initweights'

learning_rate = 0.00163
metrics_acum = {'Accuracy': [], 'Precision': [], 'Sensitivity': [], 'Specificity': [], 'F1': [], 'MCC': []}
cm_acum = []


for fold in range(0,10):
    nombre = f'{nombre0}_fold{fold}'
    txt_archivo = ""
    archivos = [4, 43, 53, 55, 63, 72, 84, 95, 105, 113, 122, 151] 
    datasets = []
    traintestval = []
    for archivo in archivos:
        txt_archivo += f"homepap-lab-full-1600{str(archivo).zfill(3)}\n"
        ds = ApneaDataset2.load_dataset(f".\data\ApneaDetection_HomePAPSignals\datasets\dataset2_archivo_1600{archivo:03d}.pth")
        if(ds._ApneaDataset2__sr != 125):
            ds.resample_segments(125)
        datasets.append(ds)
        
        train_idx = [(fold + i) % 10 for i in range(8)]
        val_idx = [(fold - 2 + i) % 10 for i in range(1)]
        test_idx = [(fold - 1 + i) % 10 for i in range(1)]
        traintestval.append([train_idx, val_idx, test_idx])

    if not os.path.exists(models_path + '/' + nombre0): 
        os.makedirs(models_path + '/' + nombre0) 

    joined_dataset, train_subsets, val_subsets, test_subsets = ApneaDataset2.join_datasets(datasets, traintestval)

    analisis_datos = joined_dataset.analisis_datos()
    analisis_datos += f"\nTrain: subsets {train_subsets}\nVal: subset {val_subsets}\nTest: subset {test_subsets}\n"
    joined_dataset.undersample_majority_class(0.0, train_subsets + val_subsets, prop = 1)
    analisis_datos += "\nUNDERSAMPLING\n" + joined_dataset.analisis_datos()

    joined_trainset = joined_dataset.get_subsets(train_subsets)
    joined_valset = joined_dataset.get_subsets(val_subsets)
    joined_testset = joined_dataset.get_subsets(test_subsets)
    analisis_datos += f"\nTrain count: {len(joined_trainset)}\n\tWith apnea: {int(sum(sum((joined_trainset[:][1]).tolist(), [])))}\n\tWithout apnea: {len(joined_trainset) - int(sum(sum((joined_trainset[:][1]).tolist(), [])))}\nVal count: {len(joined_valset)}\n\tWith apnea: {int(sum(sum((joined_valset[:][1]).tolist(), [])))}\n\tWithout apnea: {len(joined_valset) - int(sum(sum((joined_valset[:][1]).tolist(), [])))}\nTest count: {len(joined_testset)}\n\tWith apnea: {int(sum(sum((joined_testset[:][1]).tolist(), [])))}\n\tWithout apnea: {len(joined_testset) - int(sum(sum((joined_testset[:][1]).tolist(), [])))}"

    input_size = joined_dataset.signal_len()
    model = Model3(input_size, nombre).to(device)
    model.apply(init_weights)
    model_arch = model.get_architecture()
    trainer = Trainer(
        model = model,
        trainset = joined_trainset,
        valset = joined_valset,
        n_epochs = 40, 
        batch_size = 32,
        loss_fn = 'CE',
        optimizer = 'Adam',
        lr = learning_rate,
        text = txt_archivo + analisis_datos + model_arch, 
        device = device)
    model = trainer.train(models_path + '/' + nombre0, verbose = True, plot = False, save_model = False, save_best_model = False)
    tester = Tester(model = model,
                    testset = joined_testset,
                    batch_size = 32,
                    device = device, 
                    best_final = 'final')
    cm, metrics = tester.evaluate(models_path + '/' + nombre0, plot = False)
    cm_acum.append(cm)
    for key in metrics_acum:
        metrics_acum[key].append(float(metrics[key]))

cm_acum = np.array(cm_acum)
cm_mean = np.mean(cm_acum, axis=0)
cm_std = np.std(cm_acum, axis=0)
metrics_mean = {key: np.mean(metrics_acum[key]) for key in metrics_acum}
metrics_std = {key: np.std(metrics_acum[key]) for key in metrics_acum}

fig, ax = plt.subplots(figsize=(13, 6))
cm_norm = cm_mean.astype('float') / cm_mean.sum(axis=1)[:, np.newaxis] * 100
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=['without apnea', 'with apnea'])
cm_display.plot(cmap='Blues', ax=ax)
for text in ax.texts:
    text.set_visible(False)
for i in range(cm_mean.shape[0]):
    for j in range(cm_mean.shape[1]):
        text = ax.text(j, i, f'{cm_mean[i, j]:.2f} ± {cm_std[i, j]:.2f}', ha='center', va='center', color='black')
ax.set_title("Confusion Matrix Final")

metric_text = (f"Accuracy: ({metrics_mean['Accuracy']*100:.2f}±{metrics_std['Accuracy']*100:.2f})%\n"
            f"Precision: ({metrics_mean['Precision']*100:.2f}±{metrics_std['Precision']*100:.2f})%\n"
            f"Sensitivity: ({metrics_mean['Sensitivity']*100:.2f}±{metrics_std['Sensitivity']*100:.2f})%\n"
            f"Specificity: ({metrics_mean['Specificity']*100:.2f}±{metrics_std['Specificity']*100:.2f})%\n"
            f"F1: ({metrics_mean['F1']:.3f}±{metrics_std['F1']:.3f})\n"
            f"MCC: ({metrics_mean['MCC']:.3f}±{metrics_std['MCC']:.3f})")
plt.gcf().text(0.1, 0.1, metric_text, ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

if os.path.exists(models_path):
    if not os.path.exists(models_path + '/' + nombre0): 
        os.makedirs(models_path + '/' + nombre0) 
    PATH = models_path + '/' + nombre0 + '/' + nombre0 + '_cm_metrics_mean_final.png'
    plt.savefig(PATH)
plt.close()