from Imports import *
from Models import Model

class Tester:
    """Class to test the model with testset"""
    def __init__(self, model:Model, testset:Subset, batch_size:int=32, device = "cpu", best_final = ""):
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
            y_test_pred = y_test_pred.round()
            self.__all_preds__.extend(y_test_pred.detach().cpu().numpy().tolist())
            self.__all_labels__.extend(y_test.detach().cpu().numpy().tolist())
            # Save instances where the predictions were incorrect:
            self.__save_wrong_predictions__(X_test.cpu(), y_test.cpu(), y_test_pred.cpu())

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
            if y_test[idx][0] != y_test_pred[idx][0]:
                self.__wrong_predictions__.append([signal[0], int(y_test_pred[idx][0]), int(y_test[idx][0])])

    def __plot_wrong_predictions__(self, plot: bool):
        """
        Plots instances where the model's predictions were incorrect and saves plot as .png.
        
        Args:
        - plot (bool): If True, shows the plot of the incorrect predictions. If False the figures will not be displayed but will be saved in the 'models\nombre\nombre_wrongpreds.png' file.
        
        Returns: none.
        """

        plotter = Plotter(self.__model__.get_nombre(), self.__wrong_predictions__)
        plotter.plot_wrong_predictions(plot)

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
        # self.__plot_wrong_predictions__(plot)

        return self.__cm__, self.__metrics__

class Plotter:
    """Class to plot instances where the model's predictions were incorrect"""
    def __init__(self, model_name:str, wrong_predictions:list):
        """
        Initializes the Plotter object.
        
        Args:
            - model_name (str): The name of the model.
            - wrong_predictions (list): A list of wrong predictions made by the model.

        Returns: none.
        """

        self.__model_name = model_name
        self.__wrong_predictions = wrong_predictions
        self.max_rows = 5
        self.max_cols = 3
        self.plots_per_fig = self.max_rows * self.max_cols
        self.current_page = 0
        self.num_pages = math.ceil(len(self.__wrong_predictions) / self.plots_per_fig)
        self.fig = None

    def plot_wrong_predictions(self, plot:bool):
        """
        Plots the instances of wrong predictions.

        Args:
            - plot (bool): If True, shows the plot of the wrong predictions. If False, the plot will be saved but not displayed.
        
        Returns: none.
        """
        
        self.plot = plot
        self.fig, self.ax = plt.subplots(figsize=(13, 6))
        plt.subplots_adjust(left=0.1, bottom=0.2, right=0.9, top=0.9, wspace=0.1, hspace=0.5)
        
        # Botón de siguiente
        axnext = plt.axes([0.8, 0.05, 0.1, 0.075])
        self.bnext = Button(axnext, 'Next')
        self.bnext.on_clicked(self.__next_page__)

        # Botón de anterior
        axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
        self.bprev = Button(axprev, 'Previous')
        self.bprev.on_clicked(self.__prev_page__)

        self.__update_plot__()
        if (plot):
            plt.show()
        plt.close()

    def __update_plot__(self):
        """
        Updates the plot based on the current page.
        
        Args: none.
        
        Returns: none.
        """

        self.fig.clf()
        self.ax = self.fig.add_subplot(111)
        colores = ['blue', 'red']
        label_decoder = {0: 'without', 1: 'with'}

        start_idx = self.current_page * self.plots_per_fig
        end_idx = min(start_idx + self.plots_per_fig, len(self.__wrong_predictions))
        current_predictions = self.__wrong_predictions[start_idx:end_idx]

        for idx, data in enumerate(current_predictions, start=1):
            ax = self.fig.add_subplot(self.max_rows, self.max_cols, idx)
            signal, pred, target = data
            titulo = f'Targ: {label_decoder[target]} - Pred: {label_decoder[pred]}'
            eje_x = [i for i in range(len(signal))]
            ax.plot(eje_x, signal, color=colores[target])
            ax.set_title(titulo)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(0, len(signal))
            ax.set_ylim(-3, 3)

        self.ax.text(0.5, -0.1, f'Page {self.current_page + 1}/{self.num_pages}', ha='center', transform=self.ax.transAxes)
        self.ax.axis('off')

        # Botón de siguiente
        axnext = plt.axes([0.8, 0.05, 0.1, 0.075])
        self.bnext = Button(axnext, 'Next')
        self.bnext.on_clicked(self.__next_page__)

        # Botón de anterior
        axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
        self.bprev = Button(axprev, 'Previous')
        self.bprev.on_clicked(self.__prev_page__)


        if self.plot:
            self.fig.canvas.draw()

    def __next_page__(self, event):
        """
        Handles the event when the next button is clicked.

        Args:
            - event: event triggered by clicking the next button.

        Returns: none.
        """

        if self.current_page < self.num_pages - 1:
            self.current_page += 1
            self.__update_plot__()

    def __prev_page__(self, event):
        """
        Handles the event when the prev button is clicked.

        Args:
            - event: event triggered by clicking the prev button.

        Returns: none.
        """

        if self.current_page > 0:
            self.current_page -= 1
            self.__update_plot__()
