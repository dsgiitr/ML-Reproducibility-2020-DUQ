import torch
import matplotlib.pyplot as plt 
import numpy as np
from utils.cnn_duq import CNN_DUQ
from utils.datasets import all_datasets
from utils.cnn_duq import SoftmaxModel as CNN


input_size = 28
num_classes = 10
embedding_size = 256
learnable_length_scale = False
gamma = 0.999
length_scale = 0.1

model_FM_DUQ = CNN_DUQ(
    input_size,
    num_classes,
    embedding_size,
    learnable_length_scale,
    length_scale,
    gamma,
)

model_FM_DUQ.load_state_dict(torch.load('/content/DUQ_FM_30_FULL.pt'))
model_FM_DUQ.eval()

milestones = [10, 20]
args={'ensemble':5}
ds1 = all_datasets["FashionMNIST"]()
ds2 = all_datasets["MNIST"]()
input_size, num_classes, _ , test_dataset_FM = ds1
_ , _ , _ , test_dataset_M = ds2

ensemble_FM = [CNN(input_size, num_classes).cuda() for _ in range(args['ensemble'])]

ensemble_FM = torch.nn.ModuleList(ensemble_FM)


ensemble_FM.load_state_dict(torch.load('/content/FM_5_ensemble_30.pt'))
ensemble_FM.eval()


test_dataset_M.target_transform = lambda id: 100   #MNIST to be considered all wrong 

Data = test_dataset_FM + test_dataset_M


rejection_list = [0.1 , 0.2 , 0.3 ,0.4 , 0.5 ,0.6 , 0.7 , 0.8 , 0.9]

target = np.zeros((Data.__len__(),))


confidence_DUQ = np.zeros((Data.__len__(),))
pred_DUQ = np.zeros((Data.__len__(),))

for i in range(len(Data)):

  with torch.no_grad():
    _ , output = model_FM_DUQ((Data[i][0]).reshape(1,1,28,28))
    target[i] = Data[i][1]
    confidence_DUQ[i] , pred_DUQ[i]= output.max(1)

a  = np.concatenate((target.reshape(-1,1),pred_DUQ.reshape(-1,1),confidence_DUQ.reshape(-1,1)) , axis=1)
x  = a[a[:,-1].argsort()]


accuracy_DUQ = np.zeros((len(rejection_list),1))
rejected_DUQ = np.zeros((len(rejection_list),1))
i=0
for reject in rejection_list :
  y = x[:][int(reject*20000):]
  accuracy_DUQ[i] = ((y[:,0]==y[:,1]).sum())/((1-reject)*20000)
  rejected_DUQ[i] = reject*100
  i+=1

    
plt.plot(rejected_DUQ, accuracy_DUQ, color='blue', linewidth = 2, 
         marker='o', markerfacecolor='blue', markersize=5 , label='DUQ') 

confidence_DE = np.zeros((Data.__len__(),))
pred_DE = np.zeros((Data.__len__(),))

for i in range(len(Data)):
  with torch.no_grad():
      predictions = torch.stack([model(Data[i][0].reshape(1,1,28,28).cuda()) for model in ensemble_FM])

      mean_prediction = torch.mean(predictions.exp(), dim=0)
      pred_DE[i] = mean_prediction.max(1)[1]
      target[i] = Data[i][1]
      confidence_DE[i] = torch.sum(mean_prediction * torch.log(mean_prediction), dim=1)

a  = np.concatenate((target.reshape(-1,1),pred_DE.reshape(-1,1),confidence_DE.reshape(-1,1)) , axis=1)

x  = a[a[:,-1].argsort()]

accuracy_DE = np.zeros((len(rejection_list),1))
rejected_DE = np.zeros((len(rejection_list),1))
i=0
for reject in rejection_list :
  y = x[:][int(reject*20000):]
  accuracy_DE[i] = ((y[:,0]==y[:,1]).sum())/((1-reject)*20000) 
  rejected_DE[i] = reject*100
  i+=1



plt.plot(rejected_DE , accuracy_DE , color='orange', linewidth = 2, 
         marker='o', markerfacecolor='orange', markersize=5 , label='5-Deep Ensemble')



plt.ylim(0.4,1.01) 
plt.xlim(0,100) 

plt.xlabel('Percent of data rejected by uncertainity') 
plt.ylabel('Accuracy') 

plt.legend()

plt.show() 

