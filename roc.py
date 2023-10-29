import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, auc

# Load the data from 'test_scores.txt'
test_data = pd.read_csv('test_scores.txt', sep='\t')
test_data.info()
# Extract the true labels and predicted scores
true_labels = test_data['label']
predicted_scores = test_data['prediction']

# Calculate the ROC curve
fpr, tpr, thresholds = roc_curve(true_labels, predicted_scores)

# Calculate the AUC (Area Under the ROC Curve)
auroc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUROC = {auroc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')

# Save the ROC plot to a file (optional)
plt.savefig('roc_curve.png')

# Show the ROC plot
plt.show()

# Print the AUROC
print(f'Area Under the ROC Curve (AUROC): {auroc:.2f}')


from sklearn.metrics import confusion_matrix

# confusion matrix
cm_test = confusion_matrix(epoch, test_data, labels=[0,1])
print('cm_test:{}'.format(cm_test))