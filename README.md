# Resources:
+ README.md: this file.
+ data/davis/folds/test_fold_setting1.txt,train_fold_setting1.txt; data/davis/Y,ligands_can.txt,proteins.txt
  data/kiba/folds/test_fold_setting1.txt,train_fold_setting1.txt; data/kiba/Y,ligands_can.txt,proteins.txt
  These file were downloaded from https://github.com/hkmztrk/DeepDTA/tree/master/data
+ pretrained: models trained by the proposed framework 
###  source codes:
+ create_data.py: create data in pytorch format
+ utils.py: include TestbedDataset used by create_data.py to create data, and performance measures.
+ predict_with_pretrained_model.py: run this to predict affinity for testing data using models already trained stored at folder pretrained/
+ training.py: train a GraphDTA model.

# Step-by-step running:

## 0. Install Python libaries needed
+ Install pytorch_geometric following instruction at https://github.com/rusty1s/pytorch_geometric
+ Install rdkit: conda install -y -c conda-forge rdkit

## 1. Create data in pytorch format
Running
```sh
python create_data.py
```
This returns kiba_train.csv, kiba_test.csv, davis_train.csv, and davis_test.csv in data/ folder. These files are in turn input to create data in pytorch format,
stored at data/processed/, consisting of  kiba_train.pt, kiba_test.pt, davis_train.pt, and davis_test.pt.

## 2. Predict affinity with pretrained models
To predict affinity for testing data using models already trained stored at folder pretrained/. Running 
```sh
python predict_with_pretrained_model.py
```
This returns result.csv, containing the performance of the proposed models on the two datasets. The measures include rmse, mse, pearson, spearman, and ci.
The models include GINConvNet, GATNet, GAT_GCN, and GCNNet.

## 3. Train a prediction model
To train a model using training data. Running 

```sh
python training.py 0 0 0
```

where 
++ the first argument is for the index of the datasets, 0/1 for 'davis' or 'kiba', respectively;
++ the second argument is for the index of the models, 0/1/2/3 for GINConvNet, GATNet, GAT_GCN, or GCNNet, respectively;
++ and the third argument is for the index of the cuda, 0/1 for 'cuda:0' or 'cuda:1', respectively. 
+ Note that your actual CUDA name may vary from these, so please change the following code accordingly:
```sh
cuda_name = "cuda:0"
if len(sys.argv)>3:
    cuda_name = ["cuda:0","cuda:1"][int(sys.argv[3])]
```

This returns the model and result files for the modeling achieving the best mse for testing data throughtout the training.
For example, it returns two files model_GATNet_davis.model and result_GATNet_davis.csv when running GATNet on Davis data.