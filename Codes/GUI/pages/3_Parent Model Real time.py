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

st.header('Model variants')  
st.write("Choose between two models with:")
st.write(r"a) SmolVLMTSAD-0.5 : Uses a combination of BCE loss and cosine similarity loss.")
st.write(r"b) SmolVLMTSAD-1 : Uses only cosine similarity loss.")
model_option=st.selectbox(
    'Choose the model',
    ['SmolVLMTSAD-0.5','SmolVLMTSAD-1'],
	index=None,
	placeholder='Choose model type',
    label_visibility ='collapsed'
)

if model_option is not None:
    lt_txt=r"\alpha=0.5" if model_option == 'SmolVLMTSAD-0.5' else r"\alpha=1"
    st.write(f"Model with ${lt_txt}$ chosen")
    model_pth=''
    if model_option=='SmolVLMTSAD-0.5':
        model_pth='SmolVLMTSAD_0.5.pth'
    else:
        model_pth='SmolVLMTSAD_1.pth'

    df=pd.read_csv('test/test.csv')
    ts_arr=np.load('test/test.npy')
    ids=np.arange(len(ts_arr))+1
    st.write('Select the image ID from the dropdown to view real-time inference results')
    idx = st.selectbox(
        'Choose the data ID',
        ids,
        index=None,
        placeholder='Click to view dropdown',
        label_visibility='collapsed'
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
