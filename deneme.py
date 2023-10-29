import pandas as pd
import matplotlib.pyplot as plt

# Load the data from 'step_loss_acc_metrics.txt'
data = pd.read_csv('step_loss_acc_metrics.txt', sep='\t')
data.rename(columns={'# epoch':'Epoch'}, inplace=True)
data.info()
# Set the range of epochs you want to plot (e.g., from epoch 1 to epoch 76)
start_epoch = 1
end_epoch = 76
filtered_data = data[(data['Epoch'] >= start_epoch) & (data['Epoch'] <= end_epoch)]

# Extract columns for training and validation loss and accuracy
train_loss = filtered_data['training_loss']
val_loss = filtered_data['validation_loss']
train_acc = filtered_data['training_acc']
val_acc = filtered_data['validation_acc']

# Create subplots for loss and accuracy
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Loss vs. epoch
ax1.plot(filtered_data['Epoch'], train_loss, label='Training Loss', marker='o')
ax1.plot(filtered_data['Epoch'], val_loss, label='Validation Loss', marker='o')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Loss vs. Epoch')
ax1.legend()


ax2.plot(filtered_data['Epoch'], train_acc, label='Training Accuracy', marker='o')
ax2.plot(filtered_data['Epoch'], val_acc, label='Validation Accuracy', marker='o')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Accuracy vs. Epoch')
ax2.legend()


plt.savefig('training_validation_plots.png')


plt.show()


cm_test = confusion_matrix(epoch, test_data, labels=[0,1])
print('cm_test:{}'.format(cm_test))
