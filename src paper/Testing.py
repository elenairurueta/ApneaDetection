from Imports import *
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

class Tester:
    """Class to test the model with testset"""
    def __init__(self, model, test_loader, batch_size:int=32, device = "cpu", best_final = ""):
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
        self.__test_loader__ = test_loader
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
        self.__all_preds__.clear()
        self.__all_labels__.clear()

        # Iterate over each batch in the test_loader:
        for X_test, y_test in self.__test_loader__:
            X_test, y_test = X_test.to(self.device), y_test.to(self.device)
            # Get the model's predictions for the batch:
            y_test_pred = self.__model__(X_test)

            # Convert probabilities/logits to class predictions
            y_test_pred = torch.argmax(y_test_pred, dim=1)

            # Store predictions and true labels
            self.__all_preds__.extend(y_test_pred.detach().cpu().numpy().tolist())
            self.__all_labels__.extend(y_test.detach().cpu().numpy().tolist())            
    
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
            if(self.__best_final__ == ""):
                PATH = models_path + f'/roc_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.png'
            else:
                PATH = models_path + f'/roc_{self.__best_final__}.png'
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
            if(self.__best_final__ == ""):
                PATH = models_path + f'/cm_metrics_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.png'
            else:
                PATH = models_path + f'/cm_metrics_{self.__best_final__}.png'
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

        self.__metrics__ = {
            "Accuracy": accuracy_score(self.__all_labels__, self.__all_preds__),
            "Precision": precision_score(self.__all_labels__, self.__all_preds__),
            "Sensitivity": recall_score(self.__all_labels__, self.__all_preds__),
            "Specificity": recall_score(self.__all_labels__, self.__all_preds__, pos_label=0),
            "F1": f1_score(self.__all_labels__, self.__all_preds__),
            "MCC": matthews_corrcoef(self.__all_labels__, self.__all_preds__)
        }

    def evaluate(self, models_path, plot:bool = True):
        '''
        Tests the model, computes metrics and plots evaluation results.

        Args:
            - plot (bool): If True, shows the plots of evaluation results. If False, the figures will not be displayed but will be saved.
        
        Returns: none.
        '''

        if not os.path.exists(models_path):
            os.makedirs(models_path)

        self.test()
        self.__cm__ = confusion_matrix(self.__all_labels__, self.__all_preds__)
        self.__calculate_metrics__()
        self.__plot_roc_curve__(models_path, plot)
        self.__plot_metrics_confusion_matrix__(models_path, plot)
        # self.__plot_wrong_predictions__(plot)

        return self.__cm__, self.__metrics__

