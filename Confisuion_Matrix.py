import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data from 'test_scores.txt'
test_data = pd.read_csv('test_scores.txt', sep='\t')
test_data.info()
# Assuming you have a threshold for classifying as positive (adjust this threshold as needed)
threshold = 0.5  # You can set a different threshold if required

# Predict the binary class labels based on the threshold
predicted_labels = (test_data['prediction'] >= threshold).astype(int)

# Get the true labels
true_labels = test_data['label']

# Calculate the confusion matrix
confusion = confusion_matrix(true_labels, predicted_labels)

# Create a DataFrame for the confusion matrix
confusion_df = pd.DataFrame(confusion, index=['Actual Negative', 'Actual Positive'], columns=['Predicted Negative', 'Predicted Positive'])

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_df, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

# Save the confusion matrix plot to a file (optional)
plt.savefig('confusion_matrix.png')

# Print the confusion matrix as a table
print("Confusion Matrix:")
print(confusion_df)

# Show the confusion matrix plot
plt.show()
