import torch,warnings
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import recall_score,precision_score
import matplotlib.pyplot as plt
import numpy as np
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
        predicted_indices.append(s+np.argmax(prob[s:e+1]))

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

def get_inference(test_label,test_ts_data,model_pth):

    batch_size = 1
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 5. Instantiate the MultilabelVLMDataset
    test_dataset = KD_Dataset(
        binary_labels=test_label,
        time_series_data=test_ts_data
    )

    # 6. Instantiate the DataLoader
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True if DEVICE == 'cuda' else False
    )

    # Initialize the RNNAnomalyDetector model
    student_model = RNNAnomalyDetector(
        input_size=2,
        hidden_size=128,
        num_layers=2,
        bidirectional=True,
        dropout=0.3
    )

    state_dict=torch.load(model_pth,map_location=DEVICE)
    student_model.load_state_dict(state_dict['model_state_dict'])
    student_model.to(DEVICE)
    student_model.eval()
    try:
        for batch in test_dataloader:
            ts_batch = batch['time_series_data'].to(DEVICE)
            with torch.no_grad():
                # c. Get the student model's logits
                student_logits = student_model(ts_batch)
                # Get probabilities and predictions
                probs = torch.sigmoid(student_logits)
        probs=probs.cpu().numpy()

    except Exception as e:
        print(f"Error in test batch: {str(e)}")


    probs=np.squeeze(probs)
    test_label=np.squeeze(test_label)
    ind_p=cut_and_infer(probs,np.mean(probs))
    ind_t=np.where(test_label==1)[0]
    # print('Predicted anomalies',ind_p)
    # print('True anomalies',ind_t)
    # plot_image(ind_p,ind_t,np.squeeze(test_ts_data))
    pred=np.zeros(256,dtype=np.int8)
    pred[ind_p]=1
    # a,b=labels,pred
    a,b=squeeze_array(test_label,pred,2)
    score_r=recall_score(a,b,average='binary')
    score_p=precision_score(a,b,average='binary')
    return ind_p,ind_t,score_r,score_p
    # print(f'Recall: {score_r:.4f} Precision: {score_p:.4f}')
