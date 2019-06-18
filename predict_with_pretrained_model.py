import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from models.gat import GATNet
from models.gat_gcn import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet
from utils import *

def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(),total_preds.numpy().flatten()


datasets = ['davis','kiba']
modelings = [GINConvNet, GATNet, GAT_GCN, GCNNet]
cuda_name = "cuda:0"
print('cuda_name:', cuda_name)

TEST_BATCH_SIZE = 512

result = []
for dataset in datasets:
    processed_data_file_test = 'data/processed/' + dataset + '_test.pt'
    if (not os.path.isfile(processed_data_file_test)):
        print('please run create_data.py to prepare data in pytorch format!')
    else:
        test_data = TestbedDataset(root='data', dataset=dataset+'_test')
        test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
        for modeling in modelings:
            model_st = modeling.__name__
            print('\npredicting for ', dataset, ' using ', model_st)
            # training the model
            device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
            model = modeling().to(device)
            model_file_name = 'pretrained/model_' + model_st + '_' + dataset +  '.model'
            if os.path.isfile(model_file_name):            
                model.load_state_dict(torch.load(model_file_name))
                G,P = predicting(model, device, test_loader)
                ret = [rmse(G,P),mse(G,P),pearson(G,P),spearman(G,P),ci(G,P)]
                ret =[dataset, model_st] +  [round(e,3) for e in ret]
                result += [ ret ]
                print('dataset,model,rmse,mse,pearson,spearman,ci')
                print(ret)
            else:
                print('model is not available!')
with open('result.csv','w') as f:
    f.write('dataset,model,rmse,mse,pearson,spearman,ci\n')
    for ret in result:
        f.write(','.join(map(str,ret)) + '\n')