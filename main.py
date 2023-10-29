import torch as cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data_arr = pd.read_csv('step_loss_acc_metrics.txt',sep='\t')

data_arr.rename(columns={'# epoch':'Epoch'}, inplace=True)
data_arr.info()

epochs=[data_arr['Epoch']]
loss_train=data_arr['training_loss' ]
acc_training = data_arr['training_acc']
loss_valid=data_arr['validation_loss']
acc_valid=data_arr['validation_loss']
fig1,(ax1,ax2)=plt.subplots(1,2, figsize=(12,5))

ax1.plot(data_arr['Epoch'],loss_train,color='red', marker='o',label='training')
ax1.plot(data_arr['Epoch'],loss_valid,color='blue',marker='o', label='validation')

ax1.set_ylabel('loss')
ax1.set_xlabel('epoch')
ax1.legend()
ax1.grid(True,which='major',linestyle='--', color='lightgrey', alpha=0.5)

fig1.savefig('loss_vs_epoch.png',bbox_inches='tight', dpi=300)

plt.show()



ax2.plot(data_arr['Epoch'],acc_training,color='red', marker='o',label='training')
ax2.plot(data_arr['Epoch'],acc_valid,color='blue',marker='o', label='validation')

ax2.set_ylabel('accuracy')
ax2.set_xlabel('epoch')
ax2.legend()
ax2.grid(True,which='major',linestyle='--', color='lightgrey', alpha=0.5)



