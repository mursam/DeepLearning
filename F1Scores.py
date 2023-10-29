import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Load the data from 'test_scores.txt'
test_data = pd.read_csv('test_scores.txt', sep='\t')

# Assuming you have a threshold for classifying as positive (adjust this threshold as needed)
threshold = 0.5  # You can set a different threshold if required

# Predict the binary class labels based on the threshold
predicted_labels = (test_data['prediction'] >= threshold).astype(int)

# Get the true labels
true_labels = test_data['label']

# Calculate accuracy, precision, recall, and F1-score
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)

# Create a DataFrame for the metrics
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Value': [accuracy, precision, recall, f1]
})

# Print the metrics as a table
print("Metrics on the Test Set:")
print(metrics_df)

# Save the metrics as a CSV file (optional)
metrics_df.to_csv('test_metrics.csv', index=False)
