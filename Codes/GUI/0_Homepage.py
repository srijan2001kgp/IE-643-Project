import streamlit as st
from PIL import Image
st.set_page_config(
	page_title='TSAD IE 643',
	page_icon='!')

st.title('Vision Language Model Guided Knowledge Distillation for Time Series Anomaly Detection')

# Load and display an image
image = Image.open("fin_ts.png")
st.image(image, caption="Financial time series data", width=400)

st.header('About')

st.markdown('''We have implemented a knowledge distillation pipeline to detect anomalies in ECG data.
        We have used `SmolVLM` as the parent model and a stacked bidirectional `LSTM` as the student model.
        We have designed an interface where you can test the trained parent model and student model separately
    for a given ECG data.\n\nThis project is done for partial fulfillment of credits for the course `Deep Learning: Theory and practice`(IE643) under the guidance of
            **Professor P. Balamurugan** ,*Industrial Engineering and Operations Research, IIT Bombay*.
            ''')

st.header('Team Information')
st.markdown('''
            **Crescent Moon**\n  
            **SRIJAN DAS**, *24D0374*, PhD, *Department of Computer Science and Engineering, IIT Bombay*\n                        
            **ZAHIR KHAN**, *25D2015*, PhD, *Centre for Machine Intelligence & Data Science, IIT Bombay*
            ''')
