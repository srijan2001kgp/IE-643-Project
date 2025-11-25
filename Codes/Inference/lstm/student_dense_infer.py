import torch,os,time,gc,warnings
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score,recall_score,precision_score
import numpy as np
import pandas as pd
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
    if len(req_intrvl)>1:
        if req_intrvl[-1][0] - req_intrvl[-2][1] >10 and req_intrvl[-1][0] != req_intrvl[-1][1]:
            req_intrvl_merge.append(req_intrvl[-1])
        elif req_intrvl[-1][0] - req_intrvl[-1][1]<=10:
            req_intrvl_merge[-1]=(req_intrvl_merge[-1][0],req_intrvl[-1][1])
    if len(req_intrvl)==1:
        req_intrvl_merge.append(req_intrvl[0])

    predicted_indices=[]
    for i in range(len(req_intrvl_merge)):
        s=req_intrvl_merge[i][0]
        e=req_intrvl_merge[i][1]
        predicted_indices.extend(list(range(s, e+1)))

    return np.array(predicted_indices)

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

def main(): 
    wd=os.getcwd()
    print(wd)
    batch_size = 16
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    df=pd.read_csv(f"{wd}/Dense_data/test/test.csv")
    df=df.drop(columns=['ids'])
    lab=df.to_numpy()
    test_labels = lab
    test_ts_data = np.load(f'{wd}/Dense_data/test/test.npy').astype(np.float32)
    del df
    del lab
    # 5. Instantiate the MultilabelVLMDataset
    test_dataset = KD_Dataset(
        binary_labels=test_labels,
        time_series_data=test_ts_data
    )
    print(f"Test Dataset created with {len(test_dataset)} samples.")

    # 6. Instantiate the DataLoader
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True if DEVICE == 'cuda' else False
    )

    print(f"Test DataLoader created with batch size {test_dataloader.batch_size}.")
    # Initialize the RNNAnomalyDetector model
    student_model = RNNAnomalyDetector(
        input_size=2,
        hidden_size=128,
        num_layers=2,
        bidirectional=True,
        dropout=0.3
    )

    d_n="KD_train_1_0.5_dense"
    state_dict=torch.load(f"{d_n}/student_best.pth")
    student_model.load_state_dict(state_dict['model_state_dict'])

    # # Count trainable parameters
    total_params = sum(p.numel() for p in student_model.parameters())
    print(f"Total parameters: {total_params}")

    student_model.to(DEVICE)
    print("RNNAnomalyDetector (student_model) initialized and moved to device.")

    print("Starting evaluation...")
    all_labels = []
    all_probs = []
    num_batches = 0
    ts_data=[]
    student_model.eval()
    try:
        st_time=time.time()
        for batch in test_dataloader:
            labels = batch['labels'].to(DEVICE) # True labels for hard loss
            ts_batch = batch['time_series_data'].to(DEVICE)
            with torch.no_grad():
                # c. Get the student model's logits
                student_logits = student_model(ts_batch)
                num_batches += 1
                ts_data.append(ts_batch.cpu().numpy())
                # Get probabilities and predictions
                probs = torch.sigmoid(student_logits)
                all_labels.append(labels.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
                if num_batches % 50 ==0:
                    print(f"Processed {num_batches} batches")

    except Exception as e:
        print(f"Error in test batch: {str(e)}")   

    print(f"Finished inference in {time.time()-st_time} seconds.")
    # Concatenate all batches
    all_labels = np.vstack(all_labels)
    all_probs = np.vstack(all_probs)
    all_ts=np.vstack(ts_data)
    np.save(f'{d_n}/student_probs.npy',all_probs)
    np.save(f'{d_n}/student_labels.npy',all_labels)
    np.save(f'{d_n}/student_ts.npy',all_ts)
    #f=open(f'{d_n}/scores_student.log','w')
    total_recall=float(0)
    total_precision=float(0)
    for i in range(len(all_probs)):
        idx1=cut_and_infer(all_probs[i],1.5*np.mean(all_probs[i]))
        pred=np.zeros(256,dtype=np.int8)
        pred[idx1]=1
        #print(f"Predicted index{idx1}")
        #idx2=np.where(all_labels[i]==1)[0]
        #a=all_labels[i].reshape(-1,1)
        #b=pred.reshape(-1,1)
        #print(f"Actual index {idx2}")
        a,b=squeeze_array(all_labels[i],pred,2)
        # a=a.reshape(-1,a.shape[0])
        # b=b.reshape(-1,b.shape[0])
        score_r=recall_score(a,b,average='binary')
        score_p=precision_score(a,b,average='binary')
        #print(f'Recall score :{score_r:.6f} precision score {score_p:.6f}')
        total_recall+=score_r
        total_precision+=score_p

    print(f"Average recall (binary) {total_recall/len(all_probs):.8f}\n Average precision {total_precision/len(all_probs):.8f}")
    #f.close()

if __name__=='__main__':
    main()
    print("---------Finished computation--------")
    gc.collect()
