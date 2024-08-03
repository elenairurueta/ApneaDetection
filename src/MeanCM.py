from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from Imports import *

mean_cm_acum_best = [ 
    [[379.0, 76.0], [20.20, 14.80]],
    [[376.20, 80.80], [20.40, 15.60]],
    [[364.80, 87.20], [16.60, 19.40]],
    [[302.60, 137.40], [5.80, 27.20]],
    [[261.00, 189.00], [13.80, 22.20]], 
    [[345.00, 124.00], [11.20, 26.80]],
    [[318.80, 140.20], [8.40, 25.60]],
    [[311.20, 143.80], [11.80, 25.20]],
    [[308.00, 147.00], [10.80, 23.20]],
    [[369.00, 86.00], [12.80, 24.20]]
]
std_cm_acum_best = [ 
    [[16.94, 16.94], [1.17, 1.17]],
    [[18.12, 18.12], [3.14, 3.14]],
    [[24.61, 24.61], [3.07, 3.07]],
    [[54.06, 54.06], [3.54, 3.54]],
    [[26.13, 26.13], [1.94, 1.94]],
    [[71.28, 71.28], [4.07, 4.07]],
    [[29.75, 29.75], [2.50, 2.50]],
    [[17.74, 17.74], [2.23, 2.23]],
    [[24.31, 24.31], [2.93, 2.93]],
    [[17.85, 17.85], [2.56, 2.56]]
]

mean_metrics_best = {
    'Accuracy': [0.8060, 0.7940, 0.7860, 0.6980, 0.5840, 0.7320, 0.6980, 0.6840, 0.6760, 0.8000], 
    'Precision': [0.1680, 0.1640, 0.1860, 0.1800, 0.1040, 0.2080, 0.1580, 0.1500, 0.1380, 0.2240], 
    'Sensitivity': [0.4240, 0.4300, 0.5380, 0.8260, 0.6180, 0.7060, 0.7540, 0.6800, 0.6840, 0.6540], 
    'Specificity': [0.8320, 0.8220, 0.8060, 0.6900, 0.5780, 0.7360, 0.6940, 0.6860, 0.6760, 0.8120], 
    'F1': [0.236, 0.236, 0.274, 0.288, 0.182, 0.310, 0.260, 0.246, 0.228, 0.330], 
    'MCC': [0.172, 0.168, 0.222, 0.288, 0.104, 0.280, 0.244, 0.202, 0.190, 0.296]
    }
std_metrics_best = {
    'Accuracy': [0.0320, 0.0314, 0.0445, 0.1103, 0.0524, 0.1323, 0.0538, 0.0344, 0.0459, 0.0310], 
    'Precision': [0.0299, 0.0150, 0.0206, 0.0447, 0.0102, 0.0714, 0.0183, 0.0126, 0.0133, 0.0224], 
    'Sensitivity': [0.0350, 0.0867, 0.0875, 0.1063, 0.0508, 0.1050, 0.0703, 0.0593, 0.0845, 0.0700], 
    'Specificity': [0.0387, 0.0407, 0.0554, 0.1244, 0.0591, 0.1505, 0.0671, 0.0393, 0.0516, 0.0402], 
    'F1': [0.024, 0.021, 0.021, 0.054, 0.017, 0.077, 0.017, 0.014, 0.013, 0.023], 
    'MCC': [0.029, 0.028, 0.025, 0.048, 0.027, 0.073, 0.012, 0.021, 0.023, 0.022]
    }

mean_cm_acum_np = np.array(mean_cm_acum_best)
std_cm_acum_np = np.array(std_cm_acum_best)

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
for metric in mean_metrics_best.keys():
    mean_values = np.array(mean_metrics_best[metric])
    std_values = np.array(std_metrics_best[metric])
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

models_path = '/media/elena/Externo/models'

if os.path.exists(models_path):
    if not os.path.exists(models_path + '/FINAL'): 
        os.makedirs(models_path + '/FINAL') 
    PATH = models_path + '/FINAL/' + 'FINAL_cm_metrics_mean_best.png'
    plt.savefig(PATH)
plt.close()





# mean_cm_acum_final = [ 
#     [[323.40, 131.60], [15.60, 19.40]],
#     [[309.60, 147.40], [16.20, 19.80]],
#     [[288.20, 163.80], [10.00, 26.00]],
#     [[278.60, 161.40], [4.00, 29.00]],
#     [[268.40, 181.60], [15.60, 20.40]],
#     [[278.60, 190.40], [11.00, 27.00]],
#     [[268.60, 190.40], [6.60, 27.40]],
#     [[314.60, 140.40], [11.80, 25.20]],
#     [[304.40, 150.60], [11.60, 22.40]],
#     [[304.60, 150.40], [11.40, 25.60]]
# ]

# std_cm_acum_final = [ 
#     [[7.00, 7.00], [0.80, 0.80]],
#     [[19.33, 19.33], [1.60, 1.60]],
#     [[12.29, 12.29], [0.63, 0.63]],
#     [[13.06, 13.06], [13.06, 13.06]],
#     [[25.87, 25.87], [3.38, 3.38]],
#     [[22.97, 22.97], [2.10, 2.10]],
#     [[43.43, 43.43], [2.50, 2.50]],
#     [[17.06, 17.06], [2.86, 2.86]],
#     [[7.39, 7.39], [0.80, 0.80]],
#     [[20.31, 20.31], [2.06, 2.06]]
# ]

# mean_metrics_final = {
#     'Accuracy': [0.6980, 0.6700, 0.6420, 0.6480, 0.5940, 0.6040, 0.6000, 0.6920, 0.6680, 0.6720], 
#     'Precision': [0.1280, 0.1200, 0.1380, 0.1520, 0.1020, 0.1280, 0.1280, 0.1500, 0.1300, 0.1460], 
#     'Sensitivity': [0.5520, 0.5520, 0.7200, 0.8800, 0.5680, 0.7100, 0.8060, 0.6800, 0.6620, 0.6920], 
#     'Specificity': [0.7080, 0.6760, 0.6360, 0.6320, 0.5960, 0.5940, 0.5880, 0.6900, 0.6700, 0.6680], 
#     'F1': [0.210, 0.198, 0.232, 0.260, 0.172, 0.212, 0.222, 0.250, 0.216, 0.240], 
#     'MCC': [0.148, 0.124, 0.194, 0.270, 0.088, 0.164, 0.204, 0.208, 0.174, 0.200]
#     }
# std_metrics_final = {
#     'Accuracy': [0.0147, 0.0395, 0.0264, 0.0248, 0.0476, 0.0445, 0.0829, 0.0293, 0.0160, 0.0371], 
#     'Precision': [0.0075, 0.0179, 0.0098, 0.0098, 0.0117, 0.0133, 0.0172, 0.0063, 0.0089, 0.0102], 
#     'Sensitivity': [0.0240, 0.0453, 0.0190, 0.0424, 0.0937, 0.0573, 0.0703, 0.0756, 0.0240, 0.0578], 
#     'Specificity': [0.0147, 0.0408, 0.0287, 0.0306, 0.0589, 0.0500, 0.0945, 0.0363, 0.0167, 0.0449], 
#     'F1': [0.009, 0.025, 0.013, 0.260, 0.013, 0.015, 0.017, 0.020, 0.009, 0.012, 0.011], 
#     'MCC': [0.015, 0.036, 0.020, 0.270, 0.021, 0.031, 0.027, 0.024, 0.019, 0.014, 0.011]
#     }

# mean_cm_acum_np = np.array(mean_cm_acum_final)
# std_cm_acum_np = np.array(std_cm_acum_final)

# mean_final = np.mean(mean_cm_acum_np, axis=0)
# std_final = np.sqrt(np.mean(std_cm_acum_np**2, axis=0) + np.var(mean_cm_acum_np, axis=0))

# fig, ax = plt.subplots(figsize=(13, 6))
# cm_norm = mean_final.astype('float') / mean_final.sum(axis=1)[:, np.newaxis] * 100
# cm_display = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=['without apnea', 'with apnea'])
# cm_display.plot(cmap='Blues', ax=ax)
# for text in ax.texts:
#     text.set_visible(False)
# for i in range(mean_final.shape[0]):
#     for j in range(mean_final.shape[1]):
#         text = ax.text(j, i, f'{mean_final[i, j]:.2f} ± {std_final[i, j]:.2f}', ha='center', va='center', color='black')
# ax.set_title("Confusion Matrix Final")

# mean_metrics = {}
# std_metrics = {}
# for metric in mean_metrics_final.keys():
#     mean_values = np.array(mean_metrics_final[metric])
#     std_values = np.array(std_metrics_final[metric])
#     mean_final = np.mean(mean_values)
#     std_final = np.sqrt(np.mean(std_values**2) + np.var(mean_values))
#     mean_metrics[metric] = mean_final
#     std_metrics[metric] = std_final



# metric_text = (f"Accuracy: ({mean_metrics['Accuracy']*100:.2f}±{std_metrics['Accuracy']*100:.2f})%\n"
#             f"Precision: ({mean_metrics['Precision']*100:.2f}±{std_metrics['Precision']*100:.2f})%\n"
#             f"Sensitivity: ({mean_metrics['Sensitivity']*100:.2f}±{std_metrics['Sensitivity']*100:.2f})%\n"
#             f"Specificity: ({mean_metrics['Specificity']*100:.2f}±{std_metrics['Specificity']*100:.2f})%\n"
#             f"F1: ({mean_metrics['F1']:.3f}±{std_metrics['F1']:.3f})\n"
#             f"MCC: ({mean_metrics['MCC']:.3f}±{std_metrics['MCC']:.3f})")
# plt.gcf().text(0.1, 0.1, metric_text, ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

# models_path = '/media/elena/Externo/models'

# if os.path.exists(models_path):
#     if not os.path.exists(models_path + '/FINAL'): 
#         os.makedirs(models_path + '/FINAL') 
#     PATH = models_path + '/FINAL/' + 'FINAL_cm_metrics_mean_final.png'
#     plt.savefig(PATH)
# plt.close()