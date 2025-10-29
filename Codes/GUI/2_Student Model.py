import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import recall_score,precision_score

st.title('LSTM')
st.header('Model information')
st.markdown('''Total parameters: 530,689''')
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

def plot_image(probs,labels,ts):

    st.header('Resulting plot')
    st.write(":orange[Predicted anomaly locations]")
    ind_p=cut_and_infer(probs,np.mean(probs))
    st.write(ind_p)
    ind_t=np.where(labels==1)[0]
    st.write(":orange[True anomaly locations]")
    st.write(ind_t)
    # a1,b1=squeeze_array(labels,probs,2)
    # score_r=recall_score(a1,b1,average='binary')
    # score_p=precision_score(a1,b1,average='binary')
    # st.markdown(f"**Recall score:** {score_r:.4f}") 
    # st.markdown(f"**Precision score:** {score_p:.4f}")
    
    a=np.arange(0,256)
    fig,ax=plt.subplots(2,1,figsize=(10,10))
    col=['red','blue']
    for i in range(2):
        ax[i].plot(ts[:,i],color=col[i])
        ax[i].scatter(a[ind_t],ts[ind_t,i],color='cyan',label='true anomalies')
        ax[i].scatter(a[ind_p],ts[ind_p,i],color='magenta',label='predicted anomalies')
        ax[i].set_xlabel('Time steps')
        ax[i].set_ylabel(f'Channel {i+1} values')
        ax[i].legend()
    st.pyplot(fig)

st.write("Choose among the $3$ models with:")
st.write(r"a) $\lambda=0$ : No knowledge distillation, using only the student model.")
st.write(r"a) $\lambda=0.5$ : Knowledge distillation by using a combination of BCE loss with true labels and BCE loss with teacher model's logits.")
st.write(r"b) $\lambda=1$ : Knowledge distillation by using only BCE loss with teacher model's logits.")
model_option=option = st.selectbox(
    'Choose the model',
    [0,0.5,1],
	index=None,
	placeholder='Choose model type'
)

if model_option is not None:
	lt_txt=r"\alpha=0.5" if model_option == 0.5 else r"\alpha=1" if model_option ==1 else r"\alpha=0"
	st.write(f"Student model with ${lt_txt}$ chosen")
	labels=np.load(f'lstm_{model_option}\\student_labels_{model_option}.npy')
	ts_d=np.load(f'lstm_{model_option}\\student_ts_{model_option}.npy')
	probs=np.load(f'lstm_{model_option}\\student_probs_{model_option}.npy')
	ids=np.arange(len(labels))+1
	option = st.selectbox(
		'Choose the data ID',
		ids,
		index=None,
		placeholder='Click to view dropdown'
	)

	if option is not None:
		st.write('ID chosen is ',option)
		plot_image(probs[option-1], labels[option-1],ts_d[option-1])