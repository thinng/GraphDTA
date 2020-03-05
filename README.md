# Resources:

+ README.md: this file.
+ data/davis/folds/test_fold_setting1.txt,train_fold_setting1.txt; data/davis/Y,ligands_can.txt,proteins.txt
  data/kiba/folds/test_fold_setting1.txt,train_fold_setting1.txt; data/kiba/Y,ligands_can.txt,proteins.txt
  These file were downloaded from https://github.com/hkmztrk/DeepDTA/tree/master/data
+ pretrained: models trained by the proposed framework 

###  Source codes:
+ create_data.py: create data in pytorch format
+ utils.py: include TestbedDataset used by create_data.py to create data, and performance measures.
+ predict_with_pretrained_model.py: run this to predict affinity for testing data using models already trained stored at folder pretrained/
+ training.py: train a GraphDTA model.
+ models/ginconv.py, gat.py, gat_gcn.py, and gcn.py: proposed models GINConvNet, GATNet, GAT_GCN, and GCNNet receiving graphs as input for drugs.

# Step-by-step running:

## 0. Install Python libaries needed
+ Install pytorch_geometric following instruction at https://github.com/rusty1s/pytorch_geometric
+ Install rdkit: conda install -y -c conda-forge rdkit

## 1. Create data in pytorch format
Running
```sh
python create_data.py
```
This returns kiba_train.csv, kiba_test.csv, davis_train.csv, and davis_test.csv, saved in data/ folder. These files are in turn input to create data in pytorch format,
stored at data/processed/, consisting of  kiba_train.pt, kiba_test.pt, davis_train.pt, and davis_test.pt.

## 2. Predict affinity with pretrained models
To predict affinity for testing data using models already trained stored at folder pretrained/. Running 
```sh
python predict_with_pretrained_model.py
```
This returns result.csv, containing the performance of the proposed models on the two datasets. The measures include rmse, mse, pearson, spearman, and ci.
The models include GINConvNet, GATNet, GAT_GCN, and GCNNet.

## 3. Train a prediction model
To train a model using training data. The model is chosen if it gains the best MSE for testing data. This follows how a model was chosen in https://github.com/hkmztrk/DeepDTA. 
Running 

```sh
python training.py 0 0 0
```

where the first argument is for the index of the datasets, 0/1 for 'davis' or 'kiba', respectively;
 the second argument is for the index of the models, 0/1/2/3 for GINConvNet, GATNet, GAT_GCN, or GCNNet, respectively;
 and the third argument is for the index of the cuda, 0/1 for 'cuda:0' or 'cuda:1', respectively. 
 Note that your actual CUDA name may vary from these, so please change the following code accordingly:
```sh
cuda_name = "cuda:0"
if len(sys.argv)>3:
    cuda_name = ["cuda:0","cuda:1"][int(sys.argv[3])]
```

This returns the model and result files for the modelling achieving the best MSE for testing data throughout the training.
For example, it returns two files model_GATNet_davis.model and result_GATNet_davis.csv when running GATNet on Davis data.

## 4. Train a prediction model with validation 

In "3. Train a prediction model", a model is trained on training data and chosen when it gains the best MSE for testing data.
This follows how a model was chosen in https://github.com/hkmztrk/DeepDTA. The result by two ways of training is comparable though.

In this section, a model is trained on 80% of training data and chosen if it gains the best MSE for validation data, 
which is 20% of training data. Then the model is used to predict affinity for testing data.

Same arguments as in "3. Train a prediction model" are used. E.g., running 

```sh
python training_validation.py 0 0 0
```

This returns the model achieving the best MSE for validation data throughout the training and performance results of the model on testing data.
For example, it returns two files model_GATNet_davis.model and result_GATNet_davis.csv when running GATNet on Davis data.
