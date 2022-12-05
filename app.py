import streamlit as st
import pandas as pd
import numpy as np
import pickle



loaded_model = pickle.load(open('NewModelPic.sav', 'rb'))

st.sidebar.title("Selamat Datang!")
st.sidebar.write("Klasifikasi Tubuh Ideal Menggunakan Model Random Forest")


page1, page2, page3 = st.tabs(["Home", "Data", "Input Model"])

with page1:
    st.title("Klasifikasi Tubuh Ideal Menggunakan Model Random Forest")
    st.write("Dataset Yang digunakan adalah **BMI** dari [Kaggle](https://www.kaggle.com/datasets/yasserh/bmidataset)")
    st.write("Link repository Github : [https://github.com/AhmadSuud/BMIDeploy](https://github.com/AhmadSuud/BMIDeploy)")
    st.header("Deskripsi Data")
    st.write("""
        Dataset yang digunakan adalah dataset tentang BMI yang akan Mengklasifikasi apakah tinggi badan sama berat
        badan anda ideal dan ada 3 fitur  dan 1 kategori atau kelas yaitu :
        """)
    st.markdown("""
    <ul>
        <p>Fitur</p>
        <li>
            Kolom 1: Gander
            <p>Dikolom Gander ini ada dua jenis gander yaitu male dan fermale</p>
        </li>
        <li>
            Kolom 2: Height
            <p>Dikolom Height ini mejelaskan data tentang tinggi badan dengan format (cm) </p>
        </li>
        <li>
            Kolom 3: Weight
            <p> Dikolom Weight ini menjelaskan tentang berat badan dengan format (kg) </p>
        </li>
        <p>Kategori</p>
        <li>
            Kolom 4: Index
            <p>Dikolom index ini ada beberapa parameter yaitu :</p><br>
            <p>
            0 - Sangat Lemah <br>
            1 - Lemah <br>
            2 - Biasa <br>
            3 - Kegemukan <br>
            4 - Obesitas<br>
            5 - Obesitas Ekstrim</p>
        </li>
       
    </ul>
""", unsafe_allow_html=True)

with page2:
    st.title("Dataset BMI")
    data = pd.read_csv("https://raw.githubusercontent.com/AhmadSuud/datasetuas/main/bmi.csv")
    data.head()
    # deleteCol = data.drop(["Index"], axis=1)
    st.write(data)

with page3:
    st.title("Input Data Model")

    # membuat input
    Gander = st.number_input('Gander (Male/Fermale)')
    Height = st.number_input('Height(Cm)')
    Weight = st.number_input('Weight(Kg)')

dataInput = np.array([[Gander, Height, Weight]])
if st.button('Predict'):
    y_pred = loaded_model.predict(dataInput)
    st.success(f'Predict {y_pred[0]}')