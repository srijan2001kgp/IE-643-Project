import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoProcessor, AutoConfig
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List
import os,time
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')

def squeeze_array(GT, PRD, w):
    n = len(GT)
    ones = np.where(GT == 1)[0]
    if ones.size == 0:
        return np.array([0], dtype=int)

    intervals = [(max(0, i - w), min(n - 1, i + w)) for i in ones]
    intervals.sort(key=lambda x: x[0])
    #print(intervals)
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

      # print(intervals[i])
      # print(intervals[i+1])
      s1, e1 = intervals[i]
      s2, e2 = intervals[i+1]
      i=i+1
      if s2-e1-1>0:
        [modf_GT.append(0) for _ in range(s2-e1-1)]
        [modf_PRD.append(PRD[j]) for j in range(e1+1,s2)]
    #while end
    modf_GT.append(1)

    if sum(PRD[intervals[-1][0]:intervals[-1][1]+1])>0:
        modf_PRD.append(1)
    else:
        modf_PRD.append(0)

    if intervals[-1][1] <= n - 1:
      [modf_GT.append(0) for _ in range(n - intervals[-1][1] - 1)]
      [modf_PRD.append(PRD[i]) for i in range(intervals[-1][1]+1,n)]

    return np.array(modf_GT, dtype=int), np.array(modf_PRD, dtype=int)

def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences and time series data
    """
    # Separate the components
    pixel_values = [item.get('pixel_values') for item in batch if item.get('pixel_values') is not None]
    input_ids = [item.get('input_ids') for item in batch if item.get('input_ids') is not None]
    attention_masks = [item.get('attention_mask') for item in batch if item.get('attention_mask') is not None]
    labels = [item['labels'] for item in batch]
    # **FIX:** Separate time series data
    time_series_data = [item.get('time_series_data') for item in batch if item.get('time_series_data') is not None]

    # Stack labels
    labels = torch.stack(labels)

    # **FIX:** Initialize collated dictionary
    collated = {'labels': labels}

    # Handle pixel values
    if pixel_values:
        collated['pixel_values'] = torch.stack(pixel_values)

    # Handle input_ids and attention_mask with padding
    if input_ids:
        # Find max length
        max_len = max(ids.size(0) for ids in input_ids)

        # Pad input_ids
        padded_input_ids = []
        padded_attention_masks = []

        for i, ids in enumerate(input_ids):
            pad_len = max_len - ids.size(0)
            padded_ids = torch.cat([ids, torch.zeros(pad_len, dtype=ids.dtype)])
            padded_input_ids.append(padded_ids)

            if i < len(attention_masks):
                mask = attention_masks[i]
                padded_mask = torch.cat([mask, torch.zeros(pad_len, dtype=mask.dtype)])
                padded_attention_masks.append(padded_mask)

        collated['input_ids'] = torch.stack(padded_input_ids)
        if padded_attention_masks:
            collated['attention_mask'] = torch.stack(padded_attention_masks)

    # **FIX:** Stack time series data and add to collated dictionary
    if time_series_data:
        collated['time_series_data'] = torch.stack(time_series_data)


    return collated

class KD_Dataset(Dataset):
    """
    Dataset for multilabel classification with binary vectors and time series data
    """
    def __init__(
        self,
        image_paths: List[str],
        binary_labels: np.ndarray,  # Shape: (n_samples, n_classes)
        time_series_data: np.ndarray, # Shape: (n_samples, seq_len, num_channels)
        processor,
        max_length: int = 2048
    ):
        self.image_paths = image_paths
        self.binary_labels = torch.tensor(binary_labels, dtype=torch.float32)
        self.time_series_data = torch.tensor(time_series_data, dtype=torch.float32)
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load and process image
        image = Image.open(self.image_paths[idx]).convert('RGB')
        # Use proper SmolVLM format with <image> token
        prompt = "<image>The given image is a multivariate time series of two variable with 256 time stamps. Find out the anomalous time stamps from 0 to 255."
        # Process inputs - key changes here
        inputs = self.processor(
            images=[image],  # Pass as list
            text=prompt,
            return_tensors="pt",
            padding=True,  # Use dynamic padding instead of max_length
            truncation=False  # Disable truncation to avoid token mismatch
        )
        # If the sequence is too long, we'll handle it differently
        if inputs['input_ids'].size(1) > self.max_length:
            # Truncate manually while preserving image tokens structure
            inputs['input_ids'] = inputs['input_ids'][:, :self.max_length]
            if 'attention_mask' in inputs:
                inputs['attention_mask'] = inputs['attention_mask'][:, :self.max_length]

        # Remove batch dimension
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
          # Add binary labels
        inputs['labels'] = self.binary_labels[idx]
        # Add time series data
        inputs['time_series_data'] = self.time_series_data[idx]
        return inputs
    
class SmolVLMBinaryClassifier(nn.Module):
    """
    SmolVLM with binary classification head for multilabel tasks
    """
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        dropout_rate: float = 0.2,
        freeze_vision_encoder: bool = True,
        freeze_text_encoder: bool = True,
        #use_fp32: bool = True  # New parameter to control dtype
    ):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Changed from float16
            trust_remote_code=True
        )
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        # Handle different config structures for SmolVLM
        if hasattr(config, 'text_config') and hasattr(config.text_config, 'hidden_size'):
            hidden_size = config.text_config.hidden_size
        elif hasattr(config, 'hidden_size'):
            hidden_size = config.hidden_size
        elif hasattr(config, 'vocab_size'):
            # For SmolVLM, try to get from the model's actual output
            hidden_size = 576  # Updated based on your output
        else:
            hidden_size = 576  # Default fallback for SmolVLM

        print(f"Using hidden size: {hidden_size}")

        # Freeze specified components
        if freeze_vision_encoder:
            for name, param in self.base_model.named_parameters():
                if any(keyword in name.lower() for keyword in ['vision', 'visual', 'patch', 'embed']):
                    param.requires_grad = False

        if freeze_text_encoder:
            for name, param in self.base_model.named_parameters():
                if any(keyword in name.lower() for keyword in ['embed', 'layer']) and 'vision' not in name.lower():
                    param.requires_grad = False

        # SOLUTION 2: Ensure classification layers are in FP32
        # Projection layer to combine vision and text features
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        ).float()  # Explicitly set to FP32

        # Multilabel classification head
        # Replaced with LSTM
        self.lstm_classifier = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        ).float()

        self.linear_classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_classes)
        ).float()
        self.num_classes = num_classes
        #self.use_fp32 = use_fp32

    def forward(
        self,
        pixel_values=None,
        input_ids=None,
        attention_mask=None,
        **kwargs
    ):
        # Get outputs from base model
        outputs = self.base_model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        if attention_mask is not None:
            # Expand attention mask to match hidden state dimensions
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(outputs.last_hidden_state).float()
            # Apply mask and average
            sum_embeddings = (outputs.last_hidden_state * mask_expanded).sum(dim=1)
            sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
            features = sum_embeddings / sum_mask
        else:
            features = outputs.last_hidden_state.mean(dim=1)

        features = features.float()  # Ensure FP32 for downstream layers
        # Ensure features are 2D
        if features.dim() > 2:
            features = features.view(features.size(0), -1)
        # Project features
        features = self.projection(features)
        lstm_out, _ = self.lstm_classifier(features.unsqueeze(1)) # Add sequence length dimension
        # Take the output of the last time step
        lstm_out = lstm_out.squeeze(1) # Remove sequence length dimension
        # Pass through linear classifier
        logits = self.linear_classifier(lstm_out)
        return logits

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

def knowledge_distillation_loss(student_logits, teacher_logits, true_labels, temperature, alpha):

    soft_target_loss = F.binary_cross_entropy_with_logits(
    student_logits / temperature,torch.sigmoid(teacher_logits / temperature),reduction='mean')
    hard_loss = F.binary_cross_entropy_with_logits(student_logits, true_labels, reduction='mean')
    loss = (1 - alpha) * hard_loss + alpha * soft_target_loss * (temperature ** 2)
    return loss

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

def student_trainer(d_n,st_ep,best_val,student_model,teacher_model,num_epoch,lrate,warmup_steps,alpha,temp,device):
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
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps)
    #best_val=float('inf')
    st_time=time.time()
    save_path=f'{d_n}/student_best.pth'
    for epoch in range(st_ep,num_epochs):
        print(f"\n------------- Epoch {epoch+1}/{num_epochs}---------------")
        train_loss = 0
        val_loss=0
        num_batches=0
        teacher_model.eval()
        student_model.train()
        # Iterate through the train DataLoader
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        for batch in train_dataloader:
            # Get data components from the batch
            pixel_values = batch.get('pixel_values')
            input_ids = batch.get('input_ids')
            attention_mask = batch.get('attention_mask')
            labels = batch['labels'].to(device) # True labels for hard loss
            time_series_batch = batch['time_series_data'].to(device)

            # Move VLM inputs to device if they exist
            if pixel_values is not None:
                pixel_values = pixel_values.to(device)
            if input_ids is not None:
                input_ids = input_ids.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            # b. Get teacher model's logits (no_grad)
            with torch.no_grad():
                teacher_logits = teacher_model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

            # c. Get the student model's logits
            student_logits = student_model(time_series_batch)
           # d. Calculate the knowledge distillation loss
            loss = knowledge_distillation_loss(student_logits,teacher_logits,labels,temp,alpha)
            # e. Perform backpropagation
            loss.backward()

            #if (num_batches + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(student_model.parameters(),1.0)
            optimizer.step()
            optimizer.zero_grad()
                # Update learning rate
            scheduler.step()
            # g. Print the loss for the current batch
            current_lr = optimizer.param_groups[0]['lr']
            if (num_batches+1)%100==0:
                print(f'Batch {num_batches+1} \'loss\': {loss.item():.6f}, \'lr\': {current_lr:.2e}')
            train_loss += loss.item()
            num_batches+=1

        train_loss /= num_batches
        print(f"Average Training Loss: {train_loss:.6f} Time taken: {(time.time()-st_time)/3600:.2f} hrs")
        train_losses.append(train_loss)
        torch.save({'epoch': epoch,'model_state_dict': student_model.state_dict()},f'{d_n}/Epoch_{epoch+1}.pth')
        print(f"Saved model after epoch {epoch+1}")
        # validation
        student_model.eval()
        n_batch=0
        # Iterate through the validation DataLoader
        st_time=time.time()
        for batch in val_dataloader:
            # Get data components from the batch
            pixel_values = batch.get('pixel_values')
            input_ids = batch.get('input_ids')
            attention_mask = batch.get('attention_mask')
            labels = batch['labels'].to(device) # True labels for hard loss
            time_series_batch = batch['time_series_data'].to(device)

            # Move VLM inputs to device if they exist
            if pixel_values is not None:
                pixel_values = pixel_values.to(device)
            if input_ids is not None:
                input_ids = input_ids.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            with torch.no_grad():
                # b. Get teacher model's logits (no_grad)
                teacher_logits = teacher_model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                # c. Get the student model's logits
                student_logits = student_model(time_series_batch)
                # d. Calculate the knowledge distillation loss
                loss = knowledge_distillation_loss(student_logits,teacher_logits,labels,temp,alpha)
                val_loss+=loss.item()
                if (n_batch+1) % 50 ==0:
                    print(f"Batch {n_batch+1}/{len(val_dataloader)} Loss: {loss.item():.6f}")
                n_batch+=1
                
        val_loss/= n_batch
        print(f"Validation Loss: {val_loss:.6f} Time taken: {(time.time()-st_time)/60:.2f} mins")
        val_losses.append(val_loss)
        if val_loss<best_val:
            early_stop=0
            best_val=val_loss
            torch.save({'epoch': epoch,'model_state_dict': student_model.state_dict(),'val_best': best_val},save_path)
            print(f'Saved best model with validation loss {best_val:.6f}')
        else:
            early_stop+=1
            if early_stop==5:
                print("Early stopping. Getting out of training loop")
                break
    return train_losses,val_losses

# usage 
wd=os.getcwd()
batch_size = 16
NUM_EPOCH=10
WARM_UP_STEP=1000
TEACHER_MODEL_NAME = f"{wd}/models/SmolVLM-256M-Instruct"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

processor = AutoProcessor.from_pretrained(TEACHER_MODEL_NAME, trust_remote_code=True)

# Load training data
df=pd.read_csv(f"{wd}/Dense_data/train/train.csv")
# print(df.head)
ids=df['ids'].values
print(len(ids))
#ids=ids[:32]
img_paths=[]
dir_path=f"{wd}/Dense_data/train/images"
for i in ids:
    file_name=f'ecg_plot_{i}.png'
    full_path = os.path.join(dir_path, file_name)
    img_paths.append(full_path)
df_new=df.drop(columns=['ids'])
lab=df_new.to_numpy()
train_image_paths = img_paths
train_labels = lab
train_ts_data = np.load(f'{wd}/Dense_data/train/train.npy').astype(np.float32)
del df
del img_paths
del ids

# Load validation data
df=pd.read_csv(f"{wd}/Dense_data/val/val.csv")
# print(df.head)
ids=df['ids'].values
print(len(ids))
#ids=ids[:32]
img_paths=[]
dir_path=f"{wd}/Dense_data/val/images"
for i in ids:
    file_name=f'ecg_plot_{i}.png'
    full_path = os.path.join(dir_path, file_name)
    img_paths.append(full_path)
df_new=df.drop(columns=['ids'])
lab=df_new.to_numpy()
val_image_paths = img_paths
val_labels = lab
val_ts_data = np.load(f'{wd}/Dense_data/val/val.npy').astype(np.float32)
del df
del img_paths
del ids

print(len(train_image_paths))

# 5. Instantiate the MultilabelVLMDataset
train_dataset = KD_Dataset(
    image_paths=train_image_paths,
    binary_labels=train_labels,
    time_series_data=train_ts_data,
    processor=processor
)

print(f"Train Dataset created with {len(train_dataset)} samples.")

val_dataset = KD_Dataset(
    image_paths=val_image_paths,
    binary_labels=val_labels,
    time_series_data=val_ts_data,
    processor=processor
)
print(f"Validation Dataset created with {len(val_dataset)} samples.")

# 6. Instantiate the DataLoader
train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    pin_memory=True if DEVICE == 'cuda' else False,
    collate_fn=custom_collate_fn
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    pin_memory=True if DEVICE == 'cuda' else False,
    collate_fn=custom_collate_fn
)

print(f"Train DataLoader created with batch size {train_dataloader.batch_size}.")

# Ensure models are on the correct device

NUM_CLASSES = 256
teacher_model = SmolVLMBinaryClassifier(
    model_name=TEACHER_MODEL_NAME,
    num_classes=NUM_CLASSES,
    freeze_vision_encoder=True,
    freeze_text_encoder=True
)

teacher_model_path = f'smolvlmTSAD_0.5_dense/20_model.pth'
checkpoint = torch.load(teacher_model_path, map_location=DEVICE)
teacher_model.load_state_dict(checkpoint['model_state_dict'])
print("Teacher model state dict loaded successfully.")
teacher_model.to(DEVICE)

# Initialize the RNNAnomalyDetector model
student_model = RNNAnomalyDetector(
    input_size=2,
    hidden_size=128,
    num_layers=2,
    bidirectional=True,
    dropout=0.3
)
d_n=os.path.join(wd,"KD_train_1_0.5_dense")
os.makedirs(d_n,exist_ok=True)
st_ep=0
best_val=float('inf')

checkpoint_path=f"{d_n}/student_best.pth"
checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

# Load model weights
student_model.load_state_dict(checkpoint['model_state_dict'])
print("Model weights loaded.")

# Get last epoch (to continue from next one)
st_ep = checkpoint.get('epoch', 0)+1
best_val=checkpoint.get('val_best',0)
best_val=2.02999999999990
print(f"Resuming training from epoch {st_ep+1}. Best val loss {best_val:.6f}\n")

student_model.to(DEVICE)
print("RNNAnomalyDetector (student_model) initialized and moved to device.")
#summary(student_model,input_size=(batch_size,256,2))

#d_n=os.path.join(wd,"KD_train_1_0.5_dense")
#os.makedirs(d_n,exist_ok=True)
#print("Starting training...")
torch.cuda.empty_cache()
try:
    st_time=time.time()
    train_loss,val_loss=student_trainer(d_n,st_ep,best_val,student_model,teacher_model,NUM_EPOCH,2e-5,WARM_UP_STEP,1,2,DEVICE)
    print(f"\nTraining completed successfully in {(time.time()-st_time)/3600:.4f} hours")
    np.save(f"{d_n}/train_losses.npy",np.array(train_loss).astype(np.float32))
    np.save(f"{d_n}/val_losses.npy",np.array(val_loss).astype(np.float32))
except Exception as e:
    print(f"Training failed: {str(e)}")

# Load the saved loss data
#d_n = os.path.join(os.getcwd(), "KD_train_0.5")
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
