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
The code for training the *SmolVLMTSAD* model and *RNNAnomalyDetector* model are present in [Codes/Models](Codes/Models). To install the dependencies, run
```
pip install -r requirement.txt
```
  * **SmolVLMTSAD**: We need to download the **SmolVLM-Instruct** model using the Hugging Face CLI. First install `huggingface_hub` package with the CLI extras by running
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
   * **RNNAnomalyDetector**: To train *RNNAnomalyDetector* model using only the ground truth labels and without learning from *SmolVLMTSAD* model run
      ```
      python student_train.py
      ```
      The best parameters for the trained model, training losses, validation losses will be saved in a directory `LSTM_train`.

      To train *RNNAnomalyDetector* model with $\lambda=0.5$ i.e. using combined hard loss with ground truth labels and knowledge distillation loss run:
      ```
      python kd_train.py
      ```
      The best parameters for the trained model, training losses, validation losses will be saved in a directory `KD_train_0.5`. Similarly, we can do it for $\lambda=1$.
    * **Plot training and validation losses**: To plot the graphs of train and validation loss versus epochs run
      ```
      python plot_losses.py
      ```
      

## Model Inference
 * **SmolVLMTSAD**: Goto [Codes/Inference/vlm](Codes/Inference/vlm) directory for inference of the VLM model.To infer for the setting $\alpha=0.5$ run:
   ```
   python vlm_0.5_infer.py
   ```
   The model probabilities will be saved as `all_probs_0.5.npy`, the true labels as `all_labels_0.5.npy` and the time series data as `all_ts_0.5.npy`. These files will be needed for plotting the inferences.
   To infer for the setting $\alpha=1$ run:
   ```
   python vlm_1_infer.py
   ```
   The output files will be saved as before.
* **RNNAnomalyDetector**: Goto [Codes/Inference/lstm](Codes/Inference/lstm) to make inference of the student model. Set the directory of the student model in line number 183 of `student_infer.py` depending on the configuration. For $\lambda=0$, set d_n to `LSTM_train`, for $\lambda=0.5$ set d_n to `KD_train_0.5` and for $\lambda=1$ set d_n to `KD_train_1`. Now run
   ```
   python student_infer.py
   ```
   The outputs will be saved like *SmolVLMTSAD* case.
* **Inference scores**: Goto [Codes/Inference](Codes/Inference). To get the **Mean Squeezed Precision(MSP)** and **Mean Squeezed Recall(MSR)** scores run:
   ```
   python plot_inference.py
   ```
   
## GUI
First install `streamlit` in python
```
pip install streamlit
```
To run the `Streamlit` application goto [Codes/GUI](Codes/GUI). Then run
```
streamlit run 0_Homepage.py
```
This command will launch a local development server and open the Streamlit application in the web browser, typically at `http://localhost:8501`. You can open the `Parent model` or `Student model` page to select time series samples and view annotated anomaly predictions.

