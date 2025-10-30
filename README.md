# VLM Guided Knowledge Distillation for TSAD


This repository contains the codes related to this project. Goto the [Codes](Codes) directory. All the codes are written in Python 3.11.13 and are executed on Linux server.

## Dataset 
The code to generate the dataset is located in [Codes/Dataset](Codes/Dataset) directory. We need the following libraries to run the code:
  * wfdb
  * matplotlib
  * numpy
  * pandas


Run the command.
```
python data_generate.py
```
This will save the images of the ECG plots in `Plots_ECG` directory. The anomalies for each data sample are collectively stored in `labels_ECG.csv`. The time series data are stored in `ECG_segments.npy`. So, there are 70,200 data samples. To divide them into train, validation and test sets, run the command.
```
python data_split.py
```
This will create a directory `Data_split`, shuffle the data randomly and create 3 sub-directories `train`,`val`,`test`. In each directory, there will be a sub-directory `images` which will contain the images of the data samples present in that directory. Along with this there will be a `.csv` file for the true anomalies and `.npy` file for the time series data named after the parent directory i.e. in `test` directory there will be `images`, test.csv and 'test.npy`.

The directory `Data_split` is compresssed to `Data_split.tar` and uploaded to Google drive.

## Model training
The code for training the *SmolVLMTSAD* model and *RNNAnomalyDetector* model are present in [Codes/Models](Codes/Models).

To train SmolVLMTSAD* with $\alpha=0.5$, run the command
```
python vlm_train_0.5.py
```
This will create a directory `vlm_0.5`. The best model parameters will be saved as `smolvlm_best.pth` in this directory. Similarly the training loss and the validation loss will be saved in `train_losses.npy` and `val_losses.npy` respectively.

## Model Inference

## GUI

