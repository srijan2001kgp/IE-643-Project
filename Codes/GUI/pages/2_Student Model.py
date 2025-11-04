import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scores import get_scores
from student_inference import get_inference

st.title('RNNAnomalyDetector')
st.header('Model information')
st.markdown('''Total parameters: 530,689''')


def plot_image(ind_p,ind_t,ts):

    a=np.arange(0,256)
    fig,ax=plt.subplots(2,1,figsize=(10,6))
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
st.write("Choose among the $3$ models with:")
st.write(r"a) $\lambda=0$ : No knowledge distillation, using only the student model.")
st.write(r"a) $\lambda=0.5$ : Knowledge distillation by using a combination of BCE loss with true labels and BCE loss with teacher model's logits.")
st.write(r"b) $\lambda=1$ : Knowledge distillation by using only BCE loss with teacher model's logits.")
model_option= st.selectbox(
    'Choose the model',
    [0,0.5,1],
	index=None,
	placeholder='Choose model type',
    label_visibility='collapsed'
)

if model_option is not None:
    lt_txt=r"\lambda=0.5" if model_option == 0.5 else r"\lambda=1" if model_option ==1 else r"\lambda=0"
    st.write(f"Student model with ${lt_txt}$ chosen")
    df=pd.read_csv('test/test.csv')
    ids=df.iloc[:,0]
    labels=df.iloc[:,1:].to_numpy()
    ts_arr=np.load('test/test.npy')[:len(labels)]
    ids=np.arange(len(labels))+1
    if model_option!=0:
        st.write("Choose parent model used for knowledge distillation:")
        pm= st.selectbox(
        'Choose the model',
        ['SmolVLMTSAD-0.5','SmolVLMTSAD-1'],
        index=None,
        placeholder='Choose model type',
        label_visibility ='collapsed')
        if pm is not None:
            st.write(f"Parent model chosen is {pm}")
            model_pth=''
            if pm=='SmolVLMTSAD-0.5':
                if model_option==0.5:
                    model_pth='kd_0.5_hyb_best.pth'
                else:
                    model_pth='kd_1_hyb_best.pth'
            else:
                if model_option==0.5:
                    model_pth='kd_0.5_only.pth'
                else:
                    model_pth='kd_1_only.pth'
            st.write('Choose the data ID:')
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
                y=ts_arr[idx]
                ts=np.expand_dims(y,0)
                y=labels[idx]
                lbl=np.expand_dims(y,0)
                ind_p,ind_t,score_r,score_p=get_inference(lbl,ts,model_pth)
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
                
    else:
        model_pth='student_best.pth'
        st.write('Choose the data ID:')
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
            y=ts_arr[idx]
            ts=np.expand_dims(y,0)
            y=labels[idx]
            lbl=np.expand_dims(y,0)
            ind_p,ind_t,score_r,score_p=get_inference(lbl,ts,model_pth)
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
