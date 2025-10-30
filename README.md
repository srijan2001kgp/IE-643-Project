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
  * SmolVLMTSAD: We need to download the **SmolVLM-Instruct** model using the Hugging Face CLI. First install `huggingface_hub` package with the CLI extras by running
     ```
     pip install huggingface_hub[cli]
     ```
     Download the model using the repository ID `HuggingFaceTB/SmolVLM-Instruct` as follows
     ```
     huggingface-cli download HuggingFaceTB/SmolVLM-Instruct
     ```
     This will create a directoty `models` in the current working directory and save all the files associated with the model in this directory. Now, we can go forward with finetuning *SmolVLMTSAD*.
     To train *SmolVLMTSAD* with $\alpha=0.5$ i.e. using combined BCE loss and Cosine similarity loss in the loss function, run the command
     ```
     python vlm_train_0.5.py
     ```
     This will create a directory `vlm_0.5`. The best model parameters will be saved as `smolvlm_best.pth` in this directory. The code will also save the model parameters learnt adter every epoch. Similarly the training loss and the validation loss will be saved as `train_losses.npy` and `val_losses.npy` respectively.
     
     Similarly, for $\alpha=1$ i.e. using only Cosine similarity loss as the loss function, run
     ```
     python vlm_train_1.py
     ```
     Like the previous case, it will save all the output files in the directory `vlm_1`.
   * RNNAnomalyDetector: To train *RNNAnomalyDetector* model using only the ground truth labels and without learning from *SmolVLMTSAD* model run
      ```
      python student_train.py
      ```
      The best parameters for the trained model, training losses, validation losses will be saved in a directory `LSTM_train`.

## Model Inference

## GUI

