import streamlit as st
import os
from PIL import Image
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import model_from_json
import pickle

model_config = pickle.load(open("model_arch.json", "rb"))
model = model_from_json(model_config)
model.load_weights('w.h5')

directory_path = os.getcwd()

image_file1 = "imf1.jpg"
image_file2 = "imf2.jpg"


def process_img(image_path):
    im = Image.open(image_path)
    im = im.resize((120, 120))
    im_arr = np.asarray(im)
    im_arr = im_arr/255.0
    im_arr = im_arr.reshape(1, 120, 120, 3)
    im_arr = im_arr.astype('float16')
    # print(im_arr.size*im_arr.itemsize)
    return im_arr


def find_pred(im):
    res = model.predict(im)
    return np.argmax(res) + 1


hide_st_stylex = """
            <style>
            #upload-an-image {display: none;}
            </style>
            """

st.markdown("<h1 style='text-align: center; color: #bd1816;'>Steel Defect Detection Web-App</h1>",
            unsafe_allow_html=True)

st.markdown("<h3 style='text-align: center;'>A Simple WebApp to demonstrate Transfer Learning Predictions on Steel Dataset</h3>",
            unsafe_allow_html=True)

image_file = st.file_uploader('', type=['jpg', 'png'])

st.markdown("<h5 style='text-align: center; margin:20px;'>Upload an image...</h5>",
            unsafe_allow_html=True)

if(image_file):
    with st.expander('Selected Image', expanded=True):
        st.markdown(hide_st_stylex, unsafe_allow_html=True)
        st.image(image_file, use_column_width='auto')

if image_file and st.button('Predict Defect'):
    image = process_img(image_file)
    pred = find_pred(image)
    st.markdown("<h4 style='text-align: center; color: #000000;'>This image is having Class-" + str(pred) + " Defect</h4>", unsafe_allow_html=True)

st.markdown("<hr style='text-align:center; height:3px;  background-color: #000000;'>",
            unsafe_allow_html=True)

st.markdown("<h4 style='text-align: center; color: #bd1816;'>Demo-1 : Image of steel</h4>",
            unsafe_allow_html=True)

st.image(image_file1, use_column_width='auto')

if st.button('Predict Defect Img1', key='1'):
    image = process_img(image_file1)
    pred = find_pred(image)
    st.markdown("<h4 style='text-align: center; color: #000000;'>This image is having Class-" + str(pred) + " Defect</h4>", unsafe_allow_html=True)

st.markdown("<hr style='text-align:center; height:3px;  background-color: #000000;'>",
            unsafe_allow_html=True)

st.markdown("<h4 style='text-align: center; color: #bd1816;'>Demo-2 : Image of steel</h4>",
            unsafe_allow_html=True)

st.image(image_file2, use_column_width='auto')

if st.button('Predict Defect Img2', key='2'):
    image = process_img(image_file2)
    pred = find_pred(image)
    st.markdown("<h4 style='text-align: center; color: #000000;'>This image is having Class-" + str(pred) + " Defect</h4>", unsafe_allow_html=True)

hide_st_style = """
            <style>
            #MainMenu {display: none;}
            footer {display: none;}
            header {visibility:hidden}
            ::-webkit-scrollbar {display: none;}
            .stApp {
            background-image: url("https://www.man-es.com/images/default-source/default-album/iron-and-steel-applications-process-industry.jpg?sfvrsn=9667eca1_4");
            background-color: #ffffff;
            height: 100vh;
            margin:auto;
            background-repeat: no-repeat;
            background-size: cover;
            position: relative;
            }
            .css-po3vlj {background-color: #bd1816; font-weight: bold;margin:auto;}
            .egzxvld2 {margin-top:1rem; padding:3rem; border: 5px solid rgba(255, 255, 255, 0.25); border-radius: 15px; background-color: rgba(255, 255, 255, 0.75);margin:auto,40px;}
            .etr89bj1 { display: block;
                        margin:auto;
                        justify-content: center;
                        align-items: center;
                        border: 3px solid green;}
            .edgvbvh9 { display: block;
                        margin:auto;
                        justify-content: center;
                        align-items: center;
                        border: 3px solid green;}
            </style>
            """

st.markdown(hide_st_style, unsafe_allow_html=True)
