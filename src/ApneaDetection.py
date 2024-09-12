from Imports import *
from DataFormatting import ApneaDataset
from Models import Model
from Training import Trainer
from Testing import Tester

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
files = [4, 43, 53, 55, 63, 72, 84, 95, 105, 113, 122, 151]
models_path = './models'

txt_file = ""
for file in files: #
    # Initialization of variables used to create mean confusion matrices and metrics for all folds
    metrics_acum_total = {'Accuracy': [], 'Precision': [], 'Sensitivity': [], 'Specificity': [], 'F1': [], 'MCC': []}
    cm_acum_total = []
    metrics_std_total = {'Accuracy': [], 'Precision': [], 'Sensitivity': [], 'Specificity': [], 'F1': [], 'MCC': []}
    cm_std_total = []
    best_metrics_acum_total = {'Accuracy': [], 'Precision': [], 'Sensitivity': [], 'Specificity': [], 'F1': [], 'MCC': []}
    best_cm_acum_total = []
    best_metrics_std_total = {'Accuracy': [], 'Precision': [], 'Sensitivity': [], 'Specificity': [], 'F1': [], 'MCC': []}
    best_cm_std_total = []

    name0 = f'model_file_1600{str(file).zfill(3)}'

    for fold in range(0,9):
        # Initialization of variables used to create mean confusion matrices and metrics for each fold
        metrics_acum = {'Accuracy': [], 'Precision': [], 'Sensitivity': [], 'Specificity': [], 'F1': [], 'MCC': []}
        cm_acum = []
        best_metrics_acum = {'Accuracy': [], 'Precision': [], 'Sensitivity': [], 'Specificity': [], 'F1': [], 'MCC': []}
        best_cm_acum = []
        name1 = name0 + f'_fold{fold}'

        if not os.path.exists(models_path + '/' + name0 + '/' + name1):
            os.makedirs(models_path + '/' + name0 + '/' + name1)

        txt_file = f"homepap-lab-full-1600{str(file).zfill(3)}\n"
        dataset = ApneaDataset.load_dataset(f".\data\ApneaDetection_HomePAPSignals\datasets\dataset2_archivo_1600{file:03d}.pth")
        data_analysis = dataset.data_analysis()

        train_subsets = [(fold + i) % 10 for i in range(8)]
        val_subsets = [(fold - 2 + i) % 10 for i in range(1)]
        test_subsets = [(fold - 1 + i) % 10 for i in range(1)]
       
        data_analysis += f"\nTrain: subsets {train_subsets}\nVal: subset {val_subsets}\nTest: subset {test_subsets}\n"
        dataset.undersample_majority_class(0.0, train_subsets + val_subsets, prop = 1)
        data_analysis += "\nUNDERSAMPLING\n" + dataset.data_analysis()

        trainset = dataset.get_subsets(train_subsets)
        valset = dataset.get_subsets(val_subsets)
        testset = dataset.get_subsets(test_subsets)

        data_analysis += f"\n\nTrain count: {len(trainset)}\n\tWith apnea: {int(sum(sum((trainset[:][1]).tolist(), [])))}\n\tWithout apnea: {len(trainset) - int(sum(sum((trainset[:][1]).tolist(), [])))}\nVal count: {len(valset)}\n\tWith apnea: {int(sum(sum((valset[:][1]).tolist(), [])))}\n\tWithout apnea: {len(valset) - int(sum(sum((valset[:][1]).tolist(), [])))}\nTest count: {len(testset)}\n\tWith apnea: {int(sum(sum((testset[:][1]).tolist(), [])))}\n\tWithout apnea: {len(testset) - int(sum(sum((testset[:][1]).tolist(), [])))}"

        # 5 runs for each fold:
        for i in range(0,5):
            name2 = name1 + f'_{i}'
            input_size = dataset.signal_len()
            model = Model(input_size, name2)
            model_arch = model.get_architecture()
            trainer = Trainer(
                model = model,
                trainset = trainset,
                valset = valset,
                n_epochs = 100,
                batch_size = 32,
                loss_fn = 'BCE',
                optimizer = 'SGD',
                lr = 0.01,
                momentum = 0,
                text = txt_file + data_analysis + model_arch,
                device = device)
            trainer.train(models_path + '/' + name0 + '/' + name1 + '/' + name2, verbose = True, plot = False, save_best_model = True)

            # test final
            tester = Tester(model = model,
                            testset = testset,
                            batch_size = 32,
                            device = device,
                            best_final = 'final')
            cm, metrics = tester.evaluate(models_path + '/' + name0 + '/' + name1 + '/' + name2, plot = False)
            cm_acum.append(cm)
            for key in metrics_acum:
                metrics_acum[key].append(float(metrics[key]))

            # test best
            best_model = Model.load_model(models_path + '/' + name0 + '/' + name1 + '/' + name2, name2, input_size, best = True).to(device)
            best_tester = Tester(model = best_model,
                            testset = testset,
                            batch_size = 32,
                            device = device,
                            best_final = 'best')
            best_cm, best_metrics = best_tester.evaluate(models_path + '/' + name0 + '/' + name1 + '/' + name2, plot = False)
            best_cm_acum.append(best_cm)
            for key in best_metrics_acum:
                best_metrics_acum[key].append(float(best_metrics[key]))

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
            if not os.path.exists(models_path + '/' + name0 + '/' + name1):
                os.makedirs(models_path + '/' + name0 + '/' + name1)
            PATH = models_path + '/' + name0  + '/' + name1 + '/' + name1 + '_cm_metrics_mean_final.png'
            plt.savefig(PATH)
        plt.close()

        best_cm_acum = np.array(best_cm_acum)
        best_cm_mean = np.mean(best_cm_acum, axis=0)
        best_cm_std = np.std(best_cm_acum, axis=0)
        best_metrics_mean = {key: np.mean(best_metrics_acum[key]) for key in best_metrics_acum}
        best_metrics_std = {key: np.std(best_metrics_acum[key]) for key in best_metrics_acum}

        fig, ax = plt.subplots(figsize=(13, 6))
        best_cm_norm = best_cm_mean.astype('float') / best_cm_mean.sum(axis=1)[:, np.newaxis] * 100
        best_cm_display = ConfusionMatrixDisplay(confusion_matrix=best_cm_norm, display_labels=['without apnea', 'with apnea'])
        best_cm_display.plot(cmap='Blues', ax=ax)
        for text in ax.texts:
            text.set_visible(False)
        for i in range(best_cm_mean.shape[0]):
            for j in range(best_cm_mean.shape[1]):
                text = ax.text(j, i, f'{best_cm_mean[i, j]:.2f} ± {best_cm_std[i, j]:.2f}', ha='center', va='center', color='black')
        ax.set_title("Confusion Matrix Best")

        best_metric_text = (f"Accuracy: ({best_metrics_mean['Accuracy']*100:.2f}±{best_metrics_std['Accuracy']*100:.2f})%\n"
                    f"Precision: ({best_metrics_mean['Precision']*100:.2f}±{best_metrics_std['Precision']*100:.2f})%\n"
                    f"Sensitivity: ({best_metrics_mean['Sensitivity']*100:.2f}±{best_metrics_std['Sensitivity']*100:.2f})%\n"
                    f"Specificity: ({best_metrics_mean['Specificity']*100:.2f}±{best_metrics_std['Specificity']*100:.2f})%\n"
                    f"F1: ({best_metrics_mean['F1']:.3f}±{best_metrics_std['F1']:.3f})\n"
                    f"MCC: ({best_metrics_mean['MCC']:.3f}±{best_metrics_std['MCC']:.3f})")
        plt.gcf().text(0.1, 0.1, best_metric_text, ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

        if os.path.exists(models_path):
            if not os.path.exists(models_path + '/' + name0 + '/' + name1):
                os.makedirs(models_path + '/' + name0 + '/' + name1)
            PATH = models_path + '/' + name0  + '/' + name1 + '/' + name1 + '_cm_metrics_mean_best.png'
        plt.close()

        cm_acum_total.append(cm_mean)
        cm_std_total.append(cm_std)
        for key in metrics_acum_total:
            metrics_acum_total[key].append(float(metrics_mean[key]))
            metrics_std_total[key].append(float(metrics_std[key]))

        best_cm_acum_total.append(best_cm_mean)
        best_cm_std_total.append(best_cm_std)
        for key in best_metrics_acum_total:
            best_metrics_acum_total[key].append(float(best_metrics_mean[key]))
            best_metrics_std_total[key].append(float(best_metrics_std[key]))
       
    mean_cm_acum_np = np.array(cm_acum_total)
    std_cm_acum_np = np.array(cm_std_total)

    mean_final = np.mean(mean_cm_acum_np, axis=0)
    std_final = np.sqrt(np.mean(std_cm_acum_np**2, axis=0) + np.var(mean_cm_acum_np, axis=0))

    fig, ax = plt.subplots(figsize=(13, 6))
    cm_norm = mean_final.astype('float') / mean_final.sum(axis=1)[:, np.newaxis] * 100
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=['without apnea', 'with apnea'])
    cm_display.plot(cmap='Blues', ax=ax)
    for text in ax.texts:
        text.set_visible(False)
    for i in range(mean_final.shape[0]):
        for j in range(mean_final.shape[1]):
            text = ax.text(j, i, f'{mean_final[i, j]:.2f} ± {std_final[i, j]:.2f}', ha='center', va='center', color='black')
    ax.set_title("Confusion Matrix Final")

    mean_metrics = {}
    std_metrics = {}
    for metric in metrics_acum_total.keys():
        mean_values = np.array(metrics_acum_total[metric])
        std_values = np.array(metrics_std_total[metric])
        mean_final = np.mean(mean_values)
        std_final = np.sqrt(np.mean(std_values**2) + np.var(mean_values))
        mean_metrics[metric] = mean_final
        std_metrics[metric] = std_final

    metric_text = (f"Accuracy: ({mean_metrics['Accuracy']*100:.2f}±{std_metrics['Accuracy']*100:.2f})%\n"
                f"Precision: ({mean_metrics['Precision']*100:.2f}±{std_metrics['Precision']*100:.2f})%\n"
                f"Sensitivity: ({mean_metrics['Sensitivity']*100:.2f}±{std_metrics['Sensitivity']*100:.2f})%\n"
                f"Specificity: ({mean_metrics['Specificity']*100:.2f}±{std_metrics['Specificity']*100:.2f})%\n"
                f"F1: ({mean_metrics['F1']:.3f}±{std_metrics['F1']:.3f})\n"
                f"MCC: ({mean_metrics['MCC']:.3f}±{std_metrics['MCC']:.3f})")
    plt.gcf().text(0.1, 0.1, metric_text, ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    if os.path.exists(models_path):
        if not os.path.exists(models_path + '/' + name0):
                os.makedirs(models_path + '/' + name0)
        PATH = models_path + '/' + name0  + '/' + f'/{name0}_cm_metrics_mean_final.png'
        plt.savefig(PATH)
    plt.close()

    mean_cm_acum_np = np.array(best_cm_acum_total)
    std_cm_acum_np = np.array(best_cm_std_total)

    mean_final = np.mean(mean_cm_acum_np, axis=0)
    std_final = np.sqrt(np.mean(std_cm_acum_np**2, axis=0) + np.var(mean_cm_acum_np, axis=0))

    fig, ax = plt.subplots(figsize=(13, 6))
    cm_norm = mean_final.astype('float') / mean_final.sum(axis=1)[:, np.newaxis] * 100
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=['without apnea', 'with apnea'])
    cm_display.plot(cmap='Blues', ax=ax)
    for text in ax.texts:
        text.set_visible(False)
    for i in range(mean_final.shape[0]):
        for j in range(mean_final.shape[1]):
            text = ax.text(j, i, f'{mean_final[i, j]:.2f} ± {std_final[i, j]:.2f}', ha='center', va='center', color='black')
    ax.set_title("Confusion Matrix Best")

    mean_metrics = {}
    std_metrics = {}
    for metric in best_metrics_acum_total.keys():
        mean_values = np.array(best_metrics_acum_total[metric])
        std_values = np.array(best_metrics_std_total[metric])
        mean_final = np.mean(mean_values)
        std_final = np.sqrt(np.mean(std_values**2) + np.var(mean_values))
        mean_metrics[metric] = mean_final
        std_metrics[metric] = std_final

    metric_text = (f"Accuracy: ({mean_metrics['Accuracy']*100:.2f}±{std_metrics['Accuracy']*100:.2f})%\n"
                f"Precision: ({mean_metrics['Precision']*100:.2f}±{std_metrics['Precision']*100:.2f})%\n"
                f"Sensitivity: ({mean_metrics['Sensitivity']*100:.2f}±{std_metrics['Sensitivity']*100:.2f})%\n"
                f"Specificity: ({mean_metrics['Specificity']*100:.2f}±{std_metrics['Specificity']*100:.2f})%\n"
                f"F1: ({mean_metrics['F1']:.3f}±{std_metrics['F1']:.3f})\n"
                f"MCC: ({mean_metrics['MCC']:.3f}±{std_metrics['MCC']:.3f})")
    plt.gcf().text(0.1, 0.1, metric_text, ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    if os.path.exists(models_path):
        if not os.path.exists(models_path + '/' + name0):
                os.makedirs(models_path + '/' + name0)
        PATH = models_path + '/' + name0  + '/' + f'/{name0}_cm_metrics_mean_best.png'
        plt.savefig(PATH)
    plt.close()