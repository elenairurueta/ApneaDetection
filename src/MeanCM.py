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
cm00 = [388, 342, 401, 374, 291]
cm01 = [67, 113, 54, 81, 164]
cm10 = [13, 9, 17, 15, 9]
cm11 = [22, 26, 18, 20, 26]
acc = [0.8367, 0.7510, 0.8551, 0.8041, 0.6469]
prec = [0.2472, 0.1871, 0.2500, 0.1980, 0.1368]
sens = [0.6286, 0.7429, 0.5143, 0.5714, 0.7429]
spec = [0.8527, 0.7516, 0.8813, 0.8220, 0.6396]
F1 = [0.3548, 0.2989, 0.3364, 0.2941, 0.2311]
MCC = [0.3215, 0.2825, 0.2878, 0.2505, 0.2021]

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
meanMCC = np.mean(MCC)
stdMCC = np.std(MCC)

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
               f"\nF1: ({meanF1*100:.2f}+-{stdF1*100:.2f})%" +
               f"\nMCC: ({meanMCC*100:.2f}+-{stdMCC*100:.2f})%" )
plt.gcf().text(0.1, 0.1, metric_text, ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

plt.show()
