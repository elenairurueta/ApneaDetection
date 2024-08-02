from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# [[cm00, cm01], [cm10, cm11]]

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
