from Imports import *

class Tester:

    def __init__(self, model, testset, batch_size=32):
        self.__model__ = model
        self.__test_loader__ = DataLoader(testset, shuffle=False, batch_size=batch_size)
        self.__cm__ = []
        self.__metrics__ = ""
        self.__all_labels__ = []
        self.__all_preds__ = []
        self.__wrong_predictions__ = []

    def test(self):
        
        for X_test, y_test in self.__test_loader__:
            y_test_pred = self.__model__(X_test)
            y_test_pred = y_test_pred.round()
            self.__all_preds__.extend(y_test_pred.cpu().detach().numpy().tolist())
            self.__all_labels__.extend(y_test.cpu().numpy().tolist())

            self.__save_wrong_predictions__(X_test, y_test, y_test_pred)

    def __save_wrong_predictions__(self, X_test, y_test, y_test_pred):
        for idx, signal in enumerate(X_test):
            if y_test[idx][0] != y_test_pred[idx][0]:
                self.__wrong_predictions__.append([signal[0], int(y_test_pred[idx][0]), int(y_test[idx][0])])

    def __plot_wrong_predictions__(self, plot: bool):
        plotter = Plotter(self.__model__.get_nombre(), self.__wrong_predictions__)
        plotter.plot_wrong_predictions(plot)

    def __plot_confusion_matrix__(self, plot:bool):
        cm_display = ConfusionMatrixDisplay(confusion_matrix=self.__cm__, display_labels=['sin apnea', 'con apnea'])
        cm_display.plot(cmap='Blues')
        plt.title("Confusion Matrix")

        if not os.path.exists(f'models/{self.__model__.get_nombre()}'):
            os.makedirs(f'models/{self.__model__.get_nombre()}')
        PATH = f'models/{self.__model__.get_nombre()}/{self.__model__.get_nombre()}_cm.png'
        plt.savefig(PATH)

        if(plot):
            plt.show()
        else:
            plt.close()

    def __plot_roc_curve__(self, plot:bool):
        fpr, tpr, thresholds = roc_curve(self.__all_labels__, self.__all_preds__)
        plt.figure()
        lw = 2
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")

        if not os.path.exists(f'models/{self.__model__.get_nombre()}'):
            os.makedirs(f'models/{self.__model__.get_nombre()}')
        PATH = f'models/{self.__model__.get_nombre()}/{self.__model__.get_nombre()}_roc.png'
        plt.savefig(PATH)

        if(plot):
            manager = plt.get_current_fig_manager()
            manager.window.state('zoomed')
            plt.show()
        else:
            plt.close()

    def __plot_metrics_confusion_matrix__(self, plot:bool):
        cm_normalized = self.__cm__.astype('float') / self.__cm__.sum(axis=1)[:, np.newaxis] * 100

        fig, ax = plt.subplots(figsize=(10, 8))
        cm_display = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=['sin apnea', 'con apnea'])
        cm_display.plot(cmap='Blues', ax=ax)

        for text in ax.texts:
            text.set_visible(False)

        for i in range(cm_normalized.shape[0]):
            for j in range(cm_normalized.shape[1]):
                text = ax.text(j, i, f'{cm_normalized[i, j]:.2f}%', ha='center', va='center', color='black')
        ax.set_title("Confusion Matrix")
        metric_text = (f"Cantidad de datos de prueba: {len(self.__test_loader__.dataset)}\n" +
                       "\n".join([f"{k}: {v}%" for k, v in self.__metrics__.items()]))
        plt.gcf().text(0.1, 0.1, metric_text, ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

        if not os.path.exists(f'models/{self.__model__.get_nombre()}'):
            os.makedirs(f'models/{self.__model__.get_nombre()}')
        PATH = f'models/{self.__model__.get_nombre()}/{self.__model__.get_nombre()}_cm_metrics.png'
        plt.savefig(PATH)

        if(plot):
            manager = plt.get_current_fig_manager()
            manager.window.state('zoomed')
            plt.show()
        else:
            plt.close()

    def __calculate_metrics__(self):
        metrics = {
            "Accuracy": accuracy_score(self.__all_labels__, self.__all_preds__) * 100,
            "Precision": precision_score(self.__all_labels__, self.__all_preds__) * 100,
            "Sensitivity_recall": recall_score(self.__all_labels__, self.__all_preds__) * 100,
            "Specificity": recall_score(self.__all_labels__, self.__all_preds__, pos_label=0) * 100,
            "F1": f1_score(self.__all_labels__, self.__all_preds__) * 100
        }

        self.__metrics__ = {k: f"{v:.2f}" for k, v in metrics.items()}

    def evaluate(self, plot:bool = True):
        self.test()
        self.__cm__ = confusion_matrix(self.__all_labels__, self.__all_preds__)
        self.__calculate_metrics__()
        self.__plot_roc_curve__(plot)
        self.__plot_metrics_confusion_matrix__(plot)
        self.__plot_wrong_predictions__(plot)

class Plotter:
    def __init__(self, model_name, wrong_predictions):
        self.__model_name = model_name
        self.__wrong_predictions = wrong_predictions
        self.max_rows = 5
        self.max_cols = 3
        self.plots_per_fig = self.max_rows * self.max_cols
        self.current_page = 0
        self.num_pages = math.ceil(len(self.__wrong_predictions) / self.plots_per_fig)
        self.fig = None

    def plot_wrong_predictions(self, plot: bool):
        self.plot = plot
        self.fig, self.ax = plt.subplots(figsize=(15, 10))
        plt.subplots_adjust(left=0.1, bottom=0.2, right=0.9, top=0.9, wspace=0.1, hspace=0.5)
        
        # Botón de siguiente
        axnext = plt.axes([0.8, 0.05, 0.1, 0.075])
        self.bnext = Button(axnext, 'Next')
        self.bnext.on_clicked(self.next_page)

        # Botón de anterior
        axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
        self.bprev = Button(axprev, 'Previous')
        self.bprev.on_clicked(self.prev_page)

        self.update_plot()
        manager = plt.get_current_fig_manager()
        manager.window.state('zoomed')
        plt.show()

    def update_plot(self):
        self.fig.clf()
        self.ax = self.fig.add_subplot(111)  # Redefinir self.ax después de limpiar la figura
        colores = ['blue', 'red']
        label_decoder = {0: 'sin', 1: 'con'}

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

        self.ax.text(0.5, -0.1, f'Página {self.current_page + 1}/{self.num_pages}', ha='center', transform=self.ax.transAxes)
        self.ax.axis('off')

        # Botón de siguiente
        axnext = plt.axes([0.8, 0.05, 0.1, 0.075])
        self.bnext = Button(axnext, 'Next')
        self.bnext.on_clicked(self.next_page)

        # Botón de anterior
        axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
        self.bprev = Button(axprev, 'Previous')
        self.bprev.on_clicked(self.prev_page)

        if not os.path.exists(f'models/{self.__model_name}'):
            os.makedirs(f'models/{self.__model_name}')
        PATH = f'models/{self.__model_name}/{self.__model_name}_wrongpreds_page{self.current_page + 1}.png'
        plt.savefig(PATH)

        if self.plot:
            self.fig.canvas.draw()

    def next_page(self, event):
        if self.current_page < self.num_pages - 1:
            self.current_page += 1
            self.update_plot()

    def prev_page(self, event):
        if self.current_page > 0:
            self.current_page -= 1
            self.update_plot()
