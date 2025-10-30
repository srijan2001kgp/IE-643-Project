import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoProcessor, AutoConfig
import numpy as np
import pandas as pd
from typing import List, Optional
import os,time
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class KD_Dataset(Dataset):
    """
    Dataset for multilabel classification with binary vectors and time series data
    """
    def __init__(
        self,
        binary_labels: np.ndarray,  # Shape: (n_samples, n_classes)
        time_series_data: np.ndarray, # Shape: (n_samples, seq_len, num_channels)
    ):
        self.binary_labels = torch.tensor(binary_labels, dtype=torch.float32)
        self.time_series_data = torch.tensor(time_series_data, dtype=torch.float32)

    def __len__(self):
        return len(self.binary_labels)

    def __getitem__(self, idx):
       
        inputs={}
        inputs['labels'] = self.binary_labels[idx]
        # Add time series data
        inputs['time_series_data'] = self.time_series_data[idx]
        return inputs

def squeeze_array(GT, PRD, w):
    n = len(GT)
    ones = np.where(GT == 1)[0]
    if ones.size == 0:
        return np.array([0], dtype=int)

    intervals = [(max(0, i - w), min(n - 1, i + w)) for i in ones]
    intervals.sort(key=lambda x: x[0])
    modf_GT = []
    modf_PRD=[]
    if intervals[0][0] > 0:
       [modf_GT.append(0) for _ in range(intervals[0][0])]
       [modf_PRD.append(PRD[i]) for i in range(intervals[0][0])]
    i=0
    while i<len(intervals)-1:
      modf_GT.append(1)
      if sum(PRD[intervals[i][0]:intervals[i][1]+1])>0:
        modf_PRD.append(1)
      else:
        modf_PRD.append(0)
      s1, e1 = intervals[i]
      s2, e2 = intervals[i+1]
      i=i+1
      if s2-e1-1>0:
        [modf_GT.append(0) for _ in range(s2-e1-1)]
        [modf_PRD.append(PRD[j]) for j in range(e1+1,s2)]
    modf_GT.append(1)
    if sum(PRD[intervals[-1][0]:intervals[-1][1]+1])>0:
        modf_PRD.append(1)
    else:
        modf_PRD.append(0)
    if intervals[-1][1] <= n - 1:
      [modf_GT.append(0) for _ in range(n - intervals[-1][1] - 1)]
      [modf_PRD.append(PRD[i]) for i in range(intervals[-1][1]+1,n)]
    return np.array(modf_GT, dtype=int), np.array(modf_PRD, dtype=int)

class RNNAnomalyDetector(nn.Module):
    def __init__(self, input_size=2, hidden_size=128, num_layers=2, bidirectional=True, dropout=0.3):
        super(RNNAnomalyDetector, self).__init__()

        self.rnn = nn.LSTM(
            input_size=input_size,      # number of features per timestep (2 channels)
            hidden_size=hidden_size,    # latent dimension
            num_layers=num_layers,      # stacked RNN layers
            batch_first=True,           # input: (batch, seq_len, input_size)
            bidirectional=bidirectional,
            dropout=dropout
        )


        self.num_directions = 2 if bidirectional else 1

        self.fc = nn.Linear(hidden_size * self.num_directions, 1) 

    def forward(self, x):
        rnn_out, _ = self.rnn(x)

        logits = self.fc(rnn_out)

        logits = logits.squeeze(-1)
        return logits

def compute_loss(student_logits,true_labels):
    hard_loss = nn.functional.binary_cross_entropy_with_logits(student_logits, true_labels, reduction='mean')
    return hard_loss
def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
        """Linear warmup and linear decay"""
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0,
                float(num_training_steps - current_step) /
                float(max(1, num_training_steps - num_warmup_steps))
            )

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
def student_trainer(student_model,num_epoch,lrate,accumulation_steps,warmup_steps,device):
    num_epochs =num_epoch 
    early_stop=0
    train_losses=[]
    val_losses=[]
    optimizer = torch.optim.AdamW(
                student_model.parameters(),
                lr=lrate,
                weight_decay=0.01,
                betas=(0.9, 0.999)
            )
    # Learning rate scheduler with warmup
    total_steps = len(train_dataloader) * num_epochs // accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps)
    best_val=float('inf')
    st_time=time.time()
    save_path=f'{d_n}/student_best.pth'
    for epoch in range(num_epochs):
        print(f"\n------------- Epoch {epoch+1}/{num_epochs}---------------")
        train_loss = 0
        val_loss=0
        num_batches=0
        student_model.train()
        # Iterate through the train DataLoader
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        for batch in train_dataloader:
            # Get data components from the batch
            labels = batch['labels'].to(device) # True labels for hard loss
            time_series_batch = batch['time_series_data'].to(device)

            # c. Get the student model's logits
            student_logits = student_model(time_series_batch)

            # d. Calculate the knowledge distillation loss
            loss = compute_loss(student_logits,labels)

            # e. Perform backpropagation
            loss.backward()

            if (num_batches + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(student_model.parameters(),1.0)
                optimizer.step()
                optimizer.zero_grad()
                # Update learning rate
                scheduler.step()
            # g. Print the loss for the current batch
            current_lr = optimizer.param_groups[0]['lr']
            if (num_batches+1)%100==0:
                print(f'Batch {num_batches+1} \'loss\': {loss.item() * accumulation_steps:.6f}, \'lr\': {current_lr:.2e}')
            train_loss += loss.item()
            num_batches+=1

        train_loss /= num_batches
        print(f"Average Training Loss: {train_loss:.6f} Time taken: {(time.time()-st_time)/60:.2f} mins")
        train_losses.append(train_loss)
        torch.save({'epoch': epoch,'model_state_dict': student_model.state_dict()},f'{d_n}/Epoch_{epoch+1}.pth')
        print(f"Saved model after epoch {epoch+1}")
        # validation
        student_model.eval()
        n_batch=0
        # Iterate through the validation DataLoader
        st_time=time.time()
        for batch in val_dataloader:
            labels = batch['labels'].to(device) # True labels for hard loss
            time_series_batch = batch['time_series_data'].to(device)
            with torch.no_grad():
                # c. Get the student model's logits
                student_logits = student_model(time_series_batch)
                # d. Calculate the knowledge distillation loss
                loss = compute_loss(student_logits,labels)
                val_loss+=loss.item()
                if (n_batch+1) % 50 ==0:
                    print(f"  Batch {n_batch+1}/{len(val_dataloader)} Loss: {loss.item():.6f}")
                n_batch+=1
                
        val_loss/= len(val_dataloader)
        print(f"Validation Loss: {val_loss:.6f} Time taken: {(time.time()-st_time):.2f} seconds")
        val_losses.append(val_loss)
        if val_loss<best_val:
            early_stop=0
            best_val=val_loss
            torch.save({'epoch': epoch,'model_state_dict': student_model.state_dict()},save_path)
            print(f'Saved best model with validation loss {best_val:.6f}')
        else:
            early_stop+=1
            if early_stop==5:
                print("Early stopping. Getting out of training loop")
                break
    return train_losses,val_losses
    
wd=os.getcwd()
print(wd)
batch_size = 16
WARM_UP_STEP=1000
NUM_EPOCH=15
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
df=pd.read_csv(f"{wd}/Data_split/train/train.csv")
# print(df.head)
df_new=df.drop(columns=['0'])
lab=df_new.to_numpy()
train_labels = lab
train_ts_data = np.load(f'{wd}/Data_split/train/train.npy').astype(np.float32)
del df
# Load validation data
df=pd.read_csv(f"{wd}/Data_split/val/val.csv")
# print(df.head)
df_new=df.drop(columns=['0'])
lab=df_new.to_numpy()
val_labels = lab
val_ts_data = np.load(f'{wd}/Data_split/val/val.npy').astype(np.float32)
del df
# 5. Instantiate the MultilabelVLMDataset
train_dataset = KD_Dataset(
    binary_labels=train_labels,
    time_series_data=train_ts_data
)
print(f"Train Dataset created with {len(train_dataset)} samples.")
val_dataset = KD_Dataset(
    binary_labels=val_labels,
    time_series_data=val_ts_data
)
print(f"Validation Dataset created with {len(val_dataset)} samples.")
# 6. Instantiate the DataLoader
train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    pin_memory=True if DEVICE == 'cuda' else False
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    pin_memory=True if DEVICE == 'cuda' else False
)
print(f"Train DataLoader created with batch size {train_dataloader.batch_size}.")
# Initialize the RNNAnomalyDetector model
student_model = RNNAnomalyDetector(
    input_size=2,
    hidden_size=128,
    num_layers=2,
    bidirectional=True,
    dropout=0.3
)

student_model.to(DEVICE)
print("RNNAnomalyDetector (student_model) initialized and moved to device.")
d_n=os.path.join(wd,"KD_train")
os.makedirs(d_n,exist_ok=True)
print("Starting training...")
torch.cuda.empty_cache()
try:
    st_time=time.time()
    train_loss,val_loss=student_trainer(student_model,NUM_EPOCH,2e-5,1,WARM_UP_STEP,DEVICE)
    print(f"\nTraining completed successfully in {(time.time()-st_time)/60:.4f} mins")
    np.save(f"{d_n}/train_losses.npy",np.array(train_loss).astype(np.float32))
    np.save(f"{d_n}/val_losses.npy",np.array(val_loss).astype(np.float32))
except Exception as e:
    print(f"Training failed: {str(e)}")

# Load the saved loss data
d_n = os.path.join(os.getcwd(), "LSTM_train")
train_losses = np.load(f"{d_n}/train_losses.npy")
val_losses = np.load(f"{d_n}/val_losses.npy")

# Plot the training and validation losses
plt.figure(figsize=(10, 6),dpi=100)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Epochs')
plt.legend()
plt.grid(True)
# Save the plot as a PDF
plt.savefig(f"{d_n}/training_validation_loss.pdf")
plt.show()

