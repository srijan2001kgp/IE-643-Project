import numpy as np
import pandas as pd
import os,time,gc
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score,recall_score,precision_score
import warnings
warnings.filterwarnings('ignore')

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


def main():

    d_n='vlm_3_bilstm_alpha1'
    all_probs=np.load(f'{d_n}/all_probs_1.npy')
    all_labels=np.load(f'{d_n}/all_labels_1.npy')
    all_ts=np.load(f'{d_n}/all_ts_1.npy')
    total_recall=float(0)
    total_precision=float(0)
    for i in range(len(all_probs)):
        idx1=cut_and_infer(all_probs[i],np.mean(all_probs[i]))
        pred=np.zeros(256,dtype=np.int8)
        pred[idx1]=1
        #print(f"Predicted index{idx1}")
        idx2=np.where(all_labels[i]==1)[0]
        #print(f"Actual index {idx2}")
        a,b=squeeze_array(all_labels[i],pred,2)
        # a=a.reshape(-1,a.shape[0])
        # b=b.reshape(-1,b.shape[0])
        score_r=recall_score(a,b,average='binary')
        score_p=precision_score(a,b,average='binary')
        print(f'Recall score :{score_r:.6f} precision score {score_p:.6f}\n')
        total_recall+=score_r
        total_precision+=score_p

    print(f"Average recall (binary) {total_recall/len(all_probs):.8f}\n Average precision {total_precision/len(all_probs):.8f}")
    """
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
if __name__=='__main__':
    main()
    gc.collect()
