import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from parent_inference import get_inference

st.title('SmolVLMTSAD')
st.header('Model information')
st.markdown('''Total parameters: 230,577,280\n\n
Trainable parameters: 9,556,096\n\n
Percentage trainable: 4.14%'''
            )

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

def plot_image(ind_p,ind_t,ts):
            a=np.arange(0,256)
            fig,ax=plt.subplots(2,1,figsize=(8,6))
            col=['red','blue']
            for i in range(2):
                ax[i].plot(ts[:,i],color=col[i],alpha=0.7)
                ax[i].scatter(a[ind_t],ts[ind_t,i],marker='*',color='#1E3A8A',alpha=1,label='true anomalies',s=50)
                ax[i].scatter(a[ind_p],ts[ind_p,i],marker='o',color='#7F1D1D',alpha=1,label='predicted anomalies',s=40)
                ax[i].set_xlabel('Time steps')
                ax[i].set_ylabel(f'Channel {i+1} values')
                ax[i].legend()
            st.pyplot(fig)
# def plot_image(probs,labels,ts):

#     st.write(":orange[Predicted anomaly locations]")
#     ind_p=cut_and_infer(probs,np.mean(probs))
#     # st.write(ind_p)
#     cols = st.columns(len(ind_p))
#     for i, c in enumerate(cols):
#         c.markdown(f"<div style='text-align:center; border:1px solid #ccc; padding:5px; border-radius:5px;'>{ind_p[i]}</div>", unsafe_allow_html=True)
#     st.write(":blue[True anomaly locations]")
#     ind_t=np.where(labels==1)[0]
#     # st.write(ind_t)
#     cols = st.columns(len(ind_t))
#     for i, c in enumerate(cols):
#         c.markdown(f"<div style='text-align:center; border:1px solid #ccc; padding:5px; border-radius:5px;'>{ind_t[i]}</div>", unsafe_allow_html=True)
#     a=np.arange(0,256)
#     st.header('Resulting plot')
#     fig,ax=plt.subplots(2,1,figsize=(10,10))
#     col=['red','blue']
#     # #1E3A8A,#7F1D1D
#     for i in range(2):
#         ax[i].plot(ts[:,i],color=col[i],alpha=0.7)
#         ax[i].scatter(a[ind_t],ts[ind_t,i],marker='*',color='magenta',alpha=1,label='true anomalies',s=80)
#         ax[i].scatter(a[ind_p],ts[ind_p,i],marker='o',color='green',alpha=1,label='predicted anomalies',s=70)
#         ax[i].set_xlabel('Time steps')
#         ax[i].set_ylabel(f'Channel {i+1} values')
#         ax[i].legend()
#     st.pyplot(fig)    


st.header('Model variants')  
st.write("Choose between two models with:")
st.write(r"a) $\alpha=0.5$ : Uses a combination of BCE loss and cosine similarity loss.")
st.write(r"b) $\alpha=1$ : Uses only cosine similarity loss.")
model_option=st.selectbox(
    'Choose the model',
    [0.5,1],
	index=None,
	placeholder='Choose model type'
)

if model_option is not None:
    lt_txt=r"\alpha=0.5" if model_option == 0.5 else r"\alpha=1"
    st.write(f"Model with ${lt_txt}$ chosen")
    model_pth=''
    if model_option==0.5:
        model_pth='SmolVLMTSAD_0.5.pth'
    else:
        model_pth='SmolVLMTSAD_1.pth'

    df=pd.read_csv('test/test.csv')
    ts_arr=np.load('test/test.npy')
    ids=np.arange(len(ts_arr))+1
    idx = st.selectbox(
        'Choose the data ID',
        ids,
        index=None,
        placeholder='Click to view dropdown'
    )

    if idx is not None:
        st.write('ID chosen is ',idx)
        idx=idx-1
        image_id=df.iloc[idx,0]
        label=df.iloc[idx,1:].to_numpy()
        label=np.expand_dims(label,axis=0)
        ts=ts_arr[idx]
        ts=np.expand_dims(ts,axis=0)
        ind_p,ind_t,score_r,score_p=get_inference(image_id,label,ts,model_pth)
        cols = st.columns(len(ind_p))
        for i, c in enumerate(cols):
            c.markdown(f"<div style='text-align:center; border:1px solid #ccc; padding:5px; border-radius:5px;'>{ind_p[i]}</div>", unsafe_allow_html=True)
        st.write(":blue[True anomaly locations]")
        cols = st.columns(len(ind_t))
        for i, c in enumerate(cols):
            c.markdown(f"<div style='text-align:center; border:1px solid #ccc; padding:5px; border-radius:5px;'>{ind_t[i]}</div>", unsafe_allow_html=True)
        st.header('Evaluation Scores')        
        plot_image(ind_p,ind_t,np.squeeze(ts))
        st.text(f'Recall score :{score_r:.6f} precision score {score_p:.6f}\n')
