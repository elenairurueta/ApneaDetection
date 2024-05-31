from Imports import *

wrong_predictions = [] #forma: señal, predicción, target

def Test(model, nombre, testset):
    test_loader = DataLoader(testset, shuffle=False, batch_size=32)
    all_preds = []
    all_labels = []
    for X_test, y_test in test_loader:
        y_test_pred = model(X_test)
        y_test_pred = y_test_pred.round()
        all_preds.extend(y_test_pred.cpu().detach().numpy().tolist())
        all_labels.extend(y_test.cpu().numpy().tolist())

        save_wrong_predictions(X_test, y_test, y_test_pred)

    cm = confusion_matrix(all_labels, all_preds)
    formatted_metrics = metrics(all_labels, all_preds)
    plot_roc_curve(all_labels, all_preds, nombre)
    plot_metrics_confusion_matrix(cm, formatted_metrics, nombre, len(testset))
    plot_wrong_predictions(nombre)

def plot_confusion_matrix(cm):
    cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['sin apnea', 'con apnea'])
    cm_display.plot(cmap ='Blues')
    plt.show()

def metrics(all_labels, all_preds):

    metrics = {
        "Accuracy": accuracy_score(all_labels, all_preds) * 100,
        "Precision": precision_score(all_labels, all_preds) * 100,
        "Sensitivity_recall": recall_score(all_labels, all_preds) * 100,
        "Specificity": recall_score(all_labels, all_preds, pos_label=0) * 100,
        "F1": f1_score(all_labels, all_preds) * 100
    }

    formatted_metrics = {k: f"{v:.2f}" for k, v in metrics.items()}
    return formatted_metrics

def plot_metrics_confusion_matrix(cm, formatted_metrics, nombre, cantdatos):

    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    fig, ax = plt.subplots(figsize=(10, 8))
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=['sin apnea', 'con apnea'])
    cm_display.plot(cmap='Blues', ax=ax)

    for text in ax.texts:
        text.set_visible(False)

    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            text = ax.text(j, i, f'{cm_normalized[i, j]:.2f}%', ha='center', va='center', color='black')
    ax.set_title("Confusion Matrix")
    metric_text = "Cantidad de datos de prueba: " + str(cantdatos) + "\n" + "\n".join([f"{k}: {v}%" for k, v in formatted_metrics.items()])
    plt.gcf().text(0.1, 0.1, metric_text, ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    manager = plt.get_current_fig_manager()
    manager.window.state('zoomed')

    PATH = 'models/'+nombre+'_cm_metrics.png'
    plt.savefig(PATH)

    plt.show()

def plot_roc_curve(all_labels, all_preds, nombre):
    fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
    plt.figure(1)
    lw = 2
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")

    PATH = 'models/'+nombre+'_roc.png'
    plt.savefig(PATH)
    plt.show()

def save_wrong_predictions(X_test, y_test, y_test_pred):
    idx = 0
    for signal in X_test:
        if(y_test[idx][0] != y_test_pred[idx][0]):
                wrong_predictions.extend([[signal[0], int(y_test_pred[idx][0]), int(y_test[idx][0])]])
        idx += 1

def plot_wrong_predictions(nombre):
    colores = ['blue', 'red']
    idx = 1
    label_decoder = {0: 'sin', 1: 'con'}
    plt.figure(3)
    for data in wrong_predictions:
        plt.subplot(math.ceil(len(wrong_predictions)/3), 3, idx)
        signal, pred, target = data
        titulo = f'Targ: {label_decoder[target]} - Pred: {label_decoder[pred]}'
        eje_x = [i for i in range(len(signal))]
        plt.plot(eje_x, signal, color=colores[target])
        plt.title(titulo)
        # plt.xlabel('t')
        # plt.ylabel('uV')
        plt.xticks([], [])
        plt.yticks([], [])
        plt.xlim(0, len(signal))
        plt.ylim(-3, 3)
        idx += 1
    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.1, 
                    hspace=0.5)
    
    manager = plt.get_current_fig_manager()
    manager.window.state('zoomed')

    PATH = 'models/'+nombre+'_wrongpreds.png'
    plt.savefig(PATH)

    plt.show() 