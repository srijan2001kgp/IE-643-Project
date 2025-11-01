import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoProcessor, AutoConfig
from PIL import Image
import numpy as np
import pandas as pd
from typing import List, Optional
import os,time,gc
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score,recall_score,precision_score
import warnings
warnings.filterwarnings('ignore')

def cut_and_infer(prob,cut_v):
    # Convert input to numpy array if it's a list
    prob = np.array(prob)
    req_intrvl=[]
    idx_above_cut=[]
    for i in range(len(prob)):
        if prob[i] > cut_v:
            idx_above_cut.append(i)

    if not idx_above_cut: # Handle case where no values are above the cut_v
        return []

    req_indx=[idx_above_cut[0]]
    for i in range(1,len(idx_above_cut)):
        if idx_above_cut[i]-idx_above_cut[i-1] > 1:
            req_indx.append(idx_above_cut[i-1])
            req_indx.append(idx_above_cut[i])
    req_indx.append(idx_above_cut[-1])
    #print(req_indx)
    req_intrvl=[(req_indx[i],req_indx[i+1]) for i in range(0,len(req_indx),2)]
    #merging intervals with a gap of 10 or less
    req_intrvl_merge=[]
    i=0
    while i <= len(req_intrvl)-2:
        if req_intrvl[i+1][0] - req_intrvl[i][1] <=10:
            req_intrvl_merge.append((req_intrvl[i][0],req_intrvl[i+1][1]))
            i=i+2
        elif req_intrvl[i][0] != req_intrvl[i][1]:  #avoiding adding (r,r) types of element
            req_intrvl_merge.append(req_intrvl[i])
            i=i+1
        else:
            i=i+1
    if req_intrvl[-1][0] - req_intrvl[-2][1] >10 and req_intrvl[-1][0] != req_intrvl[-1][1]:
        req_intrvl_merge.append(req_intrvl[-1])
    elif req_intrvl[-1][0] - req_intrvl[-1][1]<=10:
        req_intrvl_merge[-1]=(req_intrvl_merge[-1][0],req_intrvl[-1][1])

    predicted_indices=[]
    for i in range(len(req_intrvl_merge)):
        s=req_intrvl_merge[i][0]
        e=req_intrvl_merge[i][1]
        predicted_indices.append(s+np.argmax(prob[s:e+1]))

    return np.array(predicted_indices)
    
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
    
class MultilabelVLMDataset(Dataset):
    """
    Dataset for multilabel classification with binary vectors
    """
    def __init__(
        self,
        image_paths: List[str],
        binary_labels: np.ndarray,  # Shape: (n_samples, n_classes)
        time_series_data: np.ndarray, # Shape: (n_samples, seq_len, num_channels),
        processor,
        max_length: int = 2048  # Much larger to accommodate image tokens
    ):
        self.image_paths=image_paths
        self.binary_labels = torch.tensor(binary_labels, dtype=torch.float32)
        self.time_series_data = torch.tensor(time_series_data, dtype=torch.float32)
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load and process image
        image = Image.open(self.image_paths[idx]).convert('RGB')
        # print(f"Original image size: {image.size}") # Print original image size

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
        # Add time series data
        inputs['time_series_data'] = self.time_series_data[idx]
        return inputs
    
class SmolVLMTSAD(nn.Module):
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
            dropout=0.3  # LSTM dropout is applied between layers
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
    time_series_data = [item.get('time_series_data') for item in batch if item.get('time_series_data') is not None]
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
    
    if time_series_data:
        collated['time_series_data'] = torch.stack(time_series_data)

    return collated

# usage
def main():

    dir_name=os.getcwd()
    print(f"Current directory: {dir_name}")
    # Configuration
    MODEL_NAME = f"{dir_name}/models/SmolVLM-256M-Instruct"
    NUM_CLASSES = 256
    BATCH_SIZE = 16 # batch size 16
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using device: {DEVICE}")

    # Load processor
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Load training data
    df=pd.read_csv(f"{dir_name}/Data_split/test/test.csv")
    # print(df.head)
    ids=df['0'].values
    print(len(ids))
    #ids=ids[:32]
    img_paths=[]
    dir_path=f"{dir_name}/Data_split/test/images"
    for i in ids:
        file_name=f'ecg_plot_{i}.png'
        full_path = os.path.join(dir_path, file_name)
        img_paths.append(full_path)
    df_new=df.drop(columns=['0'])
    lab=df_new.to_numpy()
    test_image_paths = img_paths
    test_labels = lab
    test_ts_data = np.load(f'{dir_name}/Data_split/test/test.npy').astype(np.float32)
    del df
    del img_paths
    del ids
    del lab

    print(f"Test samples: {len(test_image_paths)}")
    # Create datasets
    test_dataset = MultilabelVLMDataset(
        image_paths=test_image_paths,
        binary_labels=test_labels,
        time_series_data=test_ts_data,
        processor=processor
    )

    # # Create data loaders with custom collate function
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        drop_last=True,
        shuffle=False,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=True if DEVICE == 'cuda' else False,
        collate_fn=custom_collate_fn  # Use custom collate function
    )

    # # Initialize model
    print("Initializing model...")
    model = SmolVLMTSAD(
        model_name=MODEL_NAME,
        num_classes=NUM_CLASSES,
        dropout_rate=0.2,
        freeze_vision_encoder=True,
        freeze_text_encoder=True,
        #use_fp32=True  # Use FP32 to avoid gradient issues
    )
    d_n="vlm_3_bilstm"
    state_dict=torch.load(f"{d_n}/smolvlm_best.pth")
    model.load_state_dict(state_dict['model_state_dict'])

    # # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")
    #ids=np.zeros(BATCH_SIZE*len(test_loader))
    # Inference
    model.to(DEVICE)
    model.eval()
    all_labels = []
    all_probs = []
    num_batches = 0
    ts_data=[]

    st_time=time.time()
    with torch.no_grad():
        for batch in test_loader:
            try:
                # Move batch to device
                pixel_values = batch.get('pixel_values')
                input_ids = batch.get('input_ids')
                attention_mask = batch.get('attention_mask')
                labels = batch['labels'].to(DEVICE)
                ts_batch=batch['time_series_data'].to(DEVICE)
    
                if pixel_values is not None:
                    pixel_values = pixel_values.to(DEVICE)
                if input_ids is not None:
                    input_ids = input_ids.to(DEVICE)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(DEVICE)

                logits = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
    
                num_batches += 1
                ts_data.append(ts_batch.cpu().numpy())
                # Get probabilities and predictions
                probs = torch.sigmoid(logits)
                all_labels.append(labels.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
                if num_batches % 25 ==0:
                    print(f"Processed {num_batches} batches")

            except Exception as e:
                print(f"Error in validation batch: {str(e)}")
                continue
        print(f"Finished inference in {(time.time()-st_time)/60} mins.")
        # Concatenate all batches
        all_labels = np.vstack(all_labels)
        all_probs = np.vstack(all_probs)
        all_ts=np.vstack(ts_data)
#        print("shape of ts_data",all_ts.shape)
        np.save(f'{d_n}/all_probs_0.5.npy',all_probs)
        np.save(f'{d_n}/all_labels_0.5.npy',all_labels)
        np.save(f'{d_n}/all_ts_0.5.npy',all_ts)
        """
        total_recall=float(0)
        total_precision=float(0)

        for i in range(len(all_probs)): 
            idx1=cut_and_infer(all_probs[i],np.mean(all_probs[i]))
            pred=np.zeros(256,dtype=np.int8)
            pred[idx1]=1
            print(f"Predicted index{idx1}")
            idx2=np.where(all_labels[i]==1)[0]
            print(f"Actual index {idx2}")
            a,b=squeeze_array(all_labels[i],pred,2)
            # a=a.reshape(-1,a.shape[0])
            # b=b.reshape(-1,b.shape[0])
            score_r=recall_score(a,b,average='binary')
            score_p=precision_score(a,b,average='binary')
            print(f'Recall score :{score_r:.6f} precision score {score_p:.6f}\n')
            total_recall+=score_r
            total_precision+=score_p

        print(f"Average recall (binary) {total_recall/len(all_probs):.8f}\n Average precision {total_precision/len(all_probs):.8f}")
        
        num_samples=2
        print(f" Plotting the inference results for {num_samples} test samples")
        for j in range(num_samples):
            x_val=np.arange(0,256)
            y=all_ts[j]
            #print("shape of y",y.shape)
            a=cut_and_infer(all_probs[j],np.mean(all_probs[j]))
            b=np.where(all_labels[j]==1)[0]
            print('Data ',str(j),' Predicted indices ',a)
            print('True indices ',b,'\n')
            fig,ax=plt.subplots(figsize=(10,6),nrows=2,ncols=1)
            for i in range(2):          
                ax[i].plot(y[:,i],color='blue',alpha=0.3)
                ax[i].scatter(x_val[a],y[a,i],color='red')
                ax[i].scatter(x_val[b],y[b,i],color='green') 
            plt.savefig(f'{d_n}/inference_result_{j}.pdf') 
            plt.close()
            
       """
if __name__ == "__main__":
    main()
    print("---------Finished computation--------")
    gc.collect()

