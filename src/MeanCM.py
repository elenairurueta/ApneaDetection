from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# [[cm00, cm01], [cm10, cm11]]

### modelo_archivos_16000_4_43_63_72_84_95_1hp1us
# cm00 = [326, 315, 324, 304, 304]
# cm01 = [130, 141, 132, 152, 152]
# cm10 = [18, 14, 14, 14, 18]
# cm11 = [16, 20, 20, 20, 16]
# acc = [0.6980, 0.6837, 0.7020, 0.6612, 0.6531]
# prec = [0.1096, 0.1242, 0.1316, 0.1163, 0.0952]
# sens = [0.4706, 0.5882, 0.5882, 0.5882, 0.4706]
# spec = [0.7149, 0.6908, 0.7105, 0.6667, 0.6667]
# F1 = [0.1778, 0.2051, 0.2151, 0.1942, 0.1584]
# MCC = [0.1031, 0.1510, 0.1641, 0.1357, 0.0735]

### modelo_archivos_16000_4_43_63_72_84_95_1hp1us_best
# cm00 = [388, 342, 401, 374, 291]
# cm01 = [67, 113, 54, 81, 164]
# cm10 = [13, 9, 17, 15, 9]
# cm11 = [22, 26, 18, 20, 26]
# acc = [0.8367, 0.7510, 0.8551, 0.8041, 0.6469]
# prec = [0.2472, 0.1871, 0.2500, 0.1980, 0.1368]
# sens = [0.6286, 0.7429, 0.5143, 0.5714, 0.7429]
# spec = [0.8527, 0.7516, 0.8813, 0.8220, 0.6396]
# F1 = [0.3548, 0.2989, 0.3364, 0.2941, 0.2311]
# MCC = [0.3215, 0.2825, 0.2878, 0.2505, 0.2021]

### modelo_archivos_16000_4_43_63_72_84_95_2hp1us
# cm00 = [350, 343, 335, 289, 322]
# cm01 = [106, 113, 121, 167, 134]
# cm10 = [22, 23, 23, 15, 19]
# cm11 = [12, 11, 11, 19, 15]
# acc = [0.7388, 0.7224, 0.7061, 0.6286, 0.6878]
# prec = [0.1017, 0.0887, 0.0833, 0.1022, 0.1007]
# sens = [0.3529, 0.3235, 0.3235, 0.5588, 0.4412]
# spec = [0.7675, 0.7522, 0.7346, 0.6338, 0.7061]
# F1 = [0.1579, 0.1392, 0.1325, 0.1727, 0.1639]
# MCC = [0.0716, 0.0443, 0.0333, 0.1008, 0.0814]

### modelo_archivos_16000_4_43_63_72_84_95_2hp1us_best
# cm00 = [363, 383, 339, 335, 374]
# cm01 = [92, 72, 116, 120, 81]
# cm10 = [20, 18, 14, 11, 18]
# cm11 = [15, 17, 21, 24, 17]
# acc = [0.7714, 0.8163, 0.7347, 0.7327, 0.7980]
# prec = [0.1402, 0.1910, 0.1533, 0.1667, 0.1735]
# sens = [0.4286, 0.4857, 0.6000, 0.6857, 0.4857]
# spec = [0.7978, 0.8418, 0.7451, 0.7363, 0.8220]
# F1 = [0.2113, 0.2742, 0.2442, 0.2682, 0.2556]
# MCC = [0.1411, 0.2187, 0.1980, 0.2386, 0.1981]

### modelo_archivos_1600004_2hp1us
# cm00 = [56, 61, 55, 55, 60]
# cm01 = [12, 7, 13, 13, 8]
# cm10 = [2, 5, 2, 2, 4]
# cm11 = [5, 2, 5, 5, 3]
# acc = [0.8133, 0.8400, 0.8000, 0.8000, 0.8400]
# prec = [0.2941, 0.2222, 0.2778, 0.2778, 0.2727]
# sens = [0.7143, 0.2857, 0.7143, 0.7143, 0.4286]
# spec = [0.8235, 0.8971, 0.8088, 0.8088, 0.8824]
# F1 = [0.4167, 0.2500, 0.4000, 0.4000, 0.3333]

## fold 0
# cm00 = [359, 299, 315, 372, 360]
# cm01 = [96, 156, 140, 83, 95]
# cm10 = [19, 13, 12, 20, 19]
# cm11 = [16, 22, 23, 15, 16]
# acc = [0.77, 0.66, 0.69, 0.79, 0.77]
# prec = [0.14, 0.12, 0.14, 0.15, 0.14]
# sens = [0.46, 0.63, 0.66, 0.43, 0.46]
# spec = [0.79, 0.66, 0.69, 0.82, 0.79]
# F1 = [0.22, 0.21, 0.23, 0.23, 0.22]
# mcc = [0.15, 0.15, 0.19, 0.16, 0.15]

## fold 1
# cm00 = [294, 370, 308, 353, 273]
# cm01 = [163, 87, 149, 104, 184]
# cm10 = [11, 22, 14, 18, 13]
# cm11 = [25, 14, 22, 18, 23]
# acc = [0.65, 0.78, 0.67, 0.75, 0.6]
# prec = [0.13, 0.14, 0.13, 0.15, 0.11]
# sens = [0.69, 0.39, 0.61, 0.50, 0.64]
# spec = [0.64, 0.81, 0.67, 0.77, 0.60]
# F1 = [0.22, 0.20, 0.21, 0.23, 0.19]
# mcc = [0.18, 0.13, 0.16, 0.16, 0.12]

## fold 2
cm00 = [300, 342, 375, 259, 306]
cm01 = [152, 110, 77, 193, 146]
cm10 = [10, 13, 18, 8, 10]
cm11 = [26, 23, 18, 28, 26]
acc = [0.67, 0.75, 0.81, 0.59, 0.68]
prec = [0.15, 0.17, 0.19, 0.13, 0.15]
sens = [0.72, 0.64, 0.50, 0.78, 0.72]
spec = [0.66, 0.76, 0.83, 0.57, 0.68]
F1 = [0.24, 0.27, 0.27, 0.22, 0.25]
mcc = [0.21, 0.23, 0.22, 0.18, 0.22]



mean00 = np.mean(cm00)
std00 = np.std(cm00)
mean01 = np.mean(cm01)
std01 = np.std(cm01)
mean10 = np.mean(cm10)
std10 = np.std(cm10)
mean11 = np.mean(cm11)
std11 = np.std(cm11)
meanacc = np.mean(acc)
stdacc = np.std(acc)
meanprec = np.mean(prec)
stdprec = np.std(prec)
meansens = np.mean(sens)
stdsens = np.std(sens)
meanspec = np.mean(spec)
stdspec = np.std(spec)
meanF1 = np.mean(F1)
stdF1 = np.std(F1)
meanMCC = np.mean(mcc)
stdMCC = np.std(mcc)

cm_mean = np.array([[mean00, mean01], [mean10, mean11]])
cm_std = np.array([[std00, std01], [std10, std11]])
fig, ax = plt.subplots(figsize=(13, 6))
cm_norm = cm_mean.astype('float') / cm_mean.sum(axis=1)[:, np.newaxis] * 100
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=['without apnea', 'with apnea'])
cm_display.plot(cmap='Blues', ax=ax)
for text in ax.texts:
    text.set_visible(False)
for i in range(cm_mean.shape[0]):
    for j in range(cm_mean.shape[1]):
        text = ax.text(j, i, f'{cm_mean[i, j]:.2f} +- {cm_std[i, j]:.2f}', ha='center', va='center', color='black')
ax.set_title("Confusion Matrix")

metric_text = (f"Accuracy: ({meanacc*100:.2f}+-{stdacc*100:.2f})%" +
               f"\nPrecision: ({meanprec*100:.2f}+-{stdprec*100:.2f})%" +
               f"\nSensitivity: ({meansens*100:.2f}+-{stdsens*100:.2f})%" +
               f"\nSpecificity: ({meanspec*100:.2f}+-{stdspec*100:.2f})%" +
               f"\nF1: ({meanF1:.3f}+-{stdF1:.3f})"
               f"\nMCC: ({meanMCC:.3f}+-{stdMCC:.3f})" )
plt.gcf().text(0.1, 0.1, metric_text, ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

plt.show()
