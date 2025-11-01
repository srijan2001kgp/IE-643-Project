import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoProcessor, AutoConfig
from PIL import Image
import numpy as np
from typing import List, Optional, Dict, Tuple
import os,time
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score
import warnings
from scipy.ndimage import convolve
warnings.filterwarnings('ignore')

class MultilabelVLMDataset(Dataset):
    """
    Dataset for multilabel classification with binary vectors
    """
    def __init__(
        self,
        image_paths: List[str],
        binary_labels: np.ndarray,  # Shape: (n_samples, n_classes)
        processor,
        max_length: int = 2048  # Much larger to accommodate image tokens
    ):
        self.image_paths=image_paths
        self.binary_labels = torch.tensor(binary_labels, dtype=torch.float32)
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load and process image
        image = Image.open(self.image_paths[idx]).convert('RGB')
        # print(f"Original image size: {image.size}") # Print original image size

        # Use proper SmolVLM format with <image> token
        prompt = "<image>The given image is a time series of two variable with 256 time stamps. Find out the anomalous time stamps from 0 to 255."

        # Process inputs - key changes here
        inputs = self.processor(
            images=[image],  # Pass as list
            text=prompt,
            return_tensors="pt",
            padding=True,  # Use dynamic padding instead of max_length
            truncation=False  # Disable truncation to avoid token mismatch
        )

        # Print processed image size (shape of pixel_values)
        # if 'pixel_values' in inputs:
            # print(f"Processed image shape (pixel_values): {inputs['pixel_values'].shape}")


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

        return inputs


class SmolVLMTSAD(nn.Module):
    """
    SmolVLM for anomaly detection tasks
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
            dropout=0.3  # LSTM dropout is applied between layers
        ).float()

        self.linear_classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_classes)
        ).float()

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
def gaussian_kernel(w):
    """
    Creates a 1D Gaussian kernel.

    Args:
        w: The window size for the kernel.

    Returns:
        A normalized 1D Gaussian kernel.
    """
    sigma = w//3
    x = np.linspace(-w, w, 2*w+1)
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    return kernel
class VLMTrainer:
    """
    Trainer for binary multilabel classification
    """
    def __init__(
        self,
        model: nn.Module,
        save_dir:str,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 2e-5,
        warmup_steps: int = 500,
        num_epochs: int = 10,
        device: str = 'cuda',
        accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        label_smoothing: float = 0.0
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.device = device
        self.accumulation_steps = accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.criterion = nn.BCELoss()
        self.dir=save_dir
        # Label smoothing
        self.label_smoothing = label_smoothing

        # Optimizer - only optimize parameters that require gradients
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if not trainable_params:
            raise ValueError("No trainable parameters found! Check your freezing settings.")

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )

        # Learning rate scheduler with warmup
        total_steps = len(train_loader) * num_epochs // accumulation_steps
        self.scheduler = self.get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        self.train_losses = []
        self.val_metrics = []

    def get_linear_schedule_with_warmup(self, optimizer, num_warmup_steps, num_training_steps):
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
    def apply_label_smoothing( self,labels_batch,w=5):
        """Apply label smoothing to binary labels and convolve with Gaussian kernel"""
        smoothed_labels_batch = []
        labels_1=labels_batch.cpu().numpy()
        for labels in labels_1:
            original_length = len(labels)
              # Determine padding size
            pad_size = w // 2

              # Pad the labels array
            padded_labels = np.pad(labels, (pad_size, pad_size), mode='constant', constant_values=0)
            knl=gaussian_kernel(w)
              # Apply convolution
            convolved_labels = convolve(padded_labels, knl)

              # Trim the convolved output to match the original labels array size and scale the values appropriately
            trimmed_convolved_labels = convolved_labels[pad_size : pad_size + original_length]
            smoothed_labels_batch.append(trimmed_convolved_labels)

        a=np.array(smoothed_labels_batch)
        return torch.tensor(a,dtype=torch.float32)
    """
    def apply_label_smoothing(self, labels):
             if self.label_smoothing > 0:
            labels = labels * (1 - self.label_smoothing) + self.label_smoothing / 2
        return labels
    """    
    def custom_loss_function(self,model_logits,true_labels,alpha=0.5):
        #print("Inside loss computation")
        #true_labels=true_labels.to(torch.float32)
        sl=self.apply_label_smoothing(true_labels,15).to(self.device)
        true_labels=true_labels.to(torch.float32)
        model_preds=torch.sigmoid(model_logits)
        try:
            norm_preds = F.normalize(model_preds, dim=1)
            norm_labels = F.normalize(sl, dim=1)
        except Exception as e:
            print(" Error in normalization:", e)
            raise

        try:
            loss_cosine = 1 - torch.sum(norm_preds * norm_labels, dim=1).mean()
        except Exception as e:
            print(" Error in cosine part:", e)
            raise
        
        try:
            loss_bce = self.criterion(model_preds,true_labels)
        except Exception as e:
            print(" Error in BCE part:", e)
            raise
        loss=alpha*loss_cosine+(1-alpha)*loss_bce
        return loss
        
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        num_batches = 0
        print(f"Training Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(self.train_loader):
            try:
                # Move batch to device
                pixel_values = batch.get('pixel_values')
                input_ids = batch.get('input_ids')
                attention_mask = batch.get('attention_mask')
                labels = batch['labels'].to(self.device)

                if pixel_values is not None:
                    pixel_values = pixel_values.to(self.device)
                if input_ids is not None:
                    input_ids = input_ids.to(self.device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                logits = self.model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                loss = self.custom_loss_function(logits, labels)
                #print("loss ",loss)
                loss = loss / self.accumulation_steps
                loss.backward()

                # Update weights every accumulation_steps
                if (batch_idx + 1) % self.accumulation_steps == 0:

                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.max_grad_norm
                    )
                    self.optimizer.step()

                    self.optimizer.zero_grad()
                    # Update learning rate
                    self.scheduler.step()

                total_loss += loss.item() * self.accumulation_steps
                num_batches += 1

                # Update progress bar
                current_lr = self.optimizer.param_groups[0]['lr']
                if (batch_idx+1)%100==0:
                    print(f'Batch {batch_idx+1} \'loss\': {loss.item() * self.accumulation_steps:.6f}, \'lr\': {current_lr:.2e}')

            except Exception as e:
                print(f"Error in batch {batch_idx}: {str(e)}")
                continue

        if num_batches == 0:
            raise ValueError("No successful batches processed!")

        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss

    @torch.no_grad()
    def evaluate(self):
        if self.val_loader is None:
            return None
        print("Inside evaluate")
        self.model.eval()
        total_loss = 0
        num_batches = 0
        print("Evaluating")
        for batch in self.val_loader:
            try:
                # Move batch to device
                pixel_values = batch.get('pixel_values')
                input_ids = batch.get('input_ids')
                attention_mask = batch.get('attention_mask')
                labels = batch['labels'].to(self.device)

                if pixel_values is not None:
                    pixel_values = pixel_values.to(self.device)
                if input_ids is not None:
                    input_ids = input_ids.to(self.device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)

                logits = self.model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                loss = self.custom_loss_function(logits, labels)
                if (num_batches+1)%50==0:
                    print(f'Batch {num_batches+1} \'loss\': {loss.item():.6f}')
                total_loss += loss.item()
                num_batches += 1
                
                # Get probabilities and predictions
                """
                probs = torch.sigmoid(logits)
                preds = (probs > threshold).float()
                """
            except Exception as e:
                print(f"Error in validation batch: {str(e)}")
                continue

        if num_batches == 0:
            return {'loss': float('inf'), 'f1_macro': 0.0}

        loss_avg = total_loss / num_batches
        self.val_metrics.append(loss_avg)
        return loss_avg

    def train(self):
        val_best=float('inf')
        for epoch in range(self.num_epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{self.num_epochs}")
            print(f"{'='*50}")
            torch.cuda.empty_cache()
            try:
                # Training
                st_time=time.time()
                train_loss = self.train_epoch(epoch)
                print(f"Average Training Loss: {train_loss:.6f} Time taken: {(time.time()-st_time)/3600:.2f} hours")
                torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'scheduler_state_dict': self.scheduler.state_dict(),
                        }, self.dir+f"/{epoch+1}_model.pth")
                print(f"Saved model after epoch {epoch+1}")
                # Validation
                if self.val_loader:
                    st_time=time.time()
                    val_loss = self.evaluate()
                    print(f"Validation Loss: {val_loss:.6f} Time taken: {(time.time()-st_time)/3600:.2f} hours")
                    # Save best model
                    if val_loss<val_best:
                        val_best = val_loss
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'scheduler_state_dict': self.scheduler.state_dict(),
                            'best_loss': val_best
                        }, self.dir+"/smolvlm_best.pth")
                        print(f"Saved best model with best loss: {val_best:.6f}")

            except Exception as e:
                print(f"Error in epoch {epoch+1}: {str(e)}")
                continue

        return self.train_losses,self.val_metrics
    
def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences
    """
    # Separate the components
    pixel_values = [item.get('pixel_values') for item in batch if item.get('pixel_values') is not None]
    input_ids = [item.get('input_ids') for item in batch if item.get('input_ids') is not None]
    attention_masks = [item.get('attention_mask') for item in batch if item.get('attention_mask') is not None]
    labels = [item['labels'] for item in batch]

    # Stack labels
    labels = torch.stack(labels)

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

    return collated

import pandas as pd

# Configuration
wd=os.getcwd()
MODEL_NAME = f"{wd}/models/SmolVLM-256M-Instruct"
NUM_CLASSES = 256
BATCH_SIZE = 16 # batch size 16
NUM_EPOCHS = 5
LEARNING_RATE = 2e-5
WARMUP_STEPS = 1000
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using device: {DEVICE}")

# Load processor
print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Load training data
df=pd.read_csv(f"{wd}/Data_split/train/train.csv")
# print(df.head)
ids=df['0'].values
#ids=ids[:16]
print(len(ids))
img_paths=[]
dir_path=f"{wd}/Data_split/train/images"
for i in ids:
    file_name=f'ecg_plot_{i}.png'
    full_path = os.path.join(dir_path, file_name)
    img_paths.append(full_path)
df_new=df.drop(columns=['0'])
lab=df_new.to_numpy()
train_image_paths = img_paths
train_labels = lab
del df
del img_paths
del ids

# Load validation data
df=pd.read_csv(f"{wd}/Data_split/val/val.csv")
# print(df.head)
ids=df['0'].values
#ids=ids[:16]
print(len(ids))
img_paths=[]
dir_path=f"{wd}/Data_split/val/images"
for i in ids:
    file_name=f'ecg_plot_{i}.png'
    full_path = os.path.join(dir_path, file_name)
    img_paths.append(full_path)
df_new=df.drop(columns=['0'])
lab=df_new.to_numpy()
val_image_paths = img_paths
val_labels = lab
del df
del img_paths
del ids
print(f"Training samples: {len(train_image_paths)}")
print(f"Validation samples: {len(val_image_paths)}")

# Create datasets
train_dataset = MultilabelVLMDataset(
    image_paths=train_image_paths,
    binary_labels=train_labels,
    processor=processor
)

val_dataset = MultilabelVLMDataset(
    image_paths=val_image_paths,
    binary_labels=val_labels,
    processor=processor
)

# # Create data loaders with custom collate function
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    drop_last=True,
    shuffle=True,
    num_workers=0,  # Set to 0 to avoid multiprocessing issues
    pin_memory=True if DEVICE == 'cuda' else False,
    collate_fn=custom_collate_fn  # Use custom collate function
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    drop_last=True,
    shuffle=False,
    num_workers=0,
    pin_memory=True if DEVICE == 'cuda' else False,
    collate_fn=custom_collate_fn  # Use custom collate function
)

torch.cuda.empty_cache()
# # Initialize model
print("Initializing model...")
model = SmolVLMTSAD(
    model_name=MODEL_NAME,
    num_classes=NUM_CLASSES,
    dropout_rate=0.2,
    freeze_vision_encoder=True,
    freeze_text_encoder=True
)
print(model)
# # Count trainable parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")

# create directory to save model results
d_n="vlm_0.5"
os.makedirs(d_n,exist_ok=True)
# # Initialize trainer
trainer = VLMTrainer(
    model=model,
    save_dir=d_n,
    train_loader=train_loader,
    val_loader=val_loader,
    learning_rate=LEARNING_RATE,
    warmup_steps=WARMUP_STEPS,
    num_epochs=NUM_EPOCHS,
    device=DEVICE,
    accumulation_steps=1,
    max_grad_norm=1.0,
    label_smoothing=0.1
)

# # Train the model
print("Starting training...")
torch.cuda.empty_cache()
try:
    st_time=time.time()
    train_losses,val_losses = trainer.train()
    print(f"\nTraining completed successfully in {(time.time()-st_time)/3600:.4f} hours")
    np.save(f"{d_n}/train_losses.npy",np.array(train_losses).astype(np.float32))
    np.save(f"{d_n}/val_losses.npy",np.array(val_losses).astype(np.float32))
except Exception as e:
    print(f"Training failed: {str(e)}")

