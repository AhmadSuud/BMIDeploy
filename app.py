import streamlit as st
import pandas as pd
import numpy as np
import pickle



loaded_model = pickle.load(open('NewModelPic.sav', 'rb'))

st.sidebar.title("Selamat Datang!")
st.sidebar.write("Klasifikasi Tubuh Ideal Menggunakan Model Random Forest")


page1, page2, page3, page4 = st.tabs(["Home", "Data","Modeling", "Input Model"])

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
            <p> 
            0 - Male <br>
            1 - Female
            </p>
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
    st.title("Data Preprocessing")

    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    data['Gender'] = encoder.fit_transform(data['Gender'])
    st.write("""
       Di Preprocessing ini mengubah Male dan Fermale Menjadi 0 dan 1 dengan cara mengencoder menggunakan Library Sklearn 
        """)
    st.write(data)

with page3:
    st.title("Modeling")
    st.write("Akurasi Dari Setiap Model yang telah dicoba")
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state=0)
    #Random Forest
    from sklearn.ensemble import RandomForestClassifier
    rfc = RandomForestClassifier(n_estimators=200, criterion='entropy', random_state=0)
    rfc.fit(X_train, y_train)
    y_pred_rfc = rfc.predict(X_test)
    from sklearn.metrics import accuracy_score
    rf = accuracy_score(y_test, y_pred_rfc)


    #Naive Bayes
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    from sklearn.metrics import accuracy_score
    nb = accuracy_score(y_test, y_pred)

    #KNN
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    from sklearn.metrics import accuracy_score
    knn = accuracy_score(y_test, y_pred)

    #Decision Tree
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier(criterion="gini")
    clf = clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    from sklearn.metrics import accuracy_score
    dt = accuracy_score(y_test, y_pred)

    #K-Means
    from sklearn.cluster import KMeans


    #Hasil Semua model
    st.write("Random Forest = ", rf*100,"%")
    st.write("Naive Bayes = ", nb*100,"%")
    st.write("KNN = ", knn*100,"%")
    st.write("Decision Tree", dt*100,"%" )
    # st.write("K-Means", km)
with page4:
    st.title("Input Data Model")

    # membuat input
    Gander = st.number_input('Gander (Male/Female)')
    Height = st.number_input('Height(Cm)')
    Weight = st.number_input('Weight(Kg)')

dataInput = np.array([[Gander, Height, Weight]])
if st.button('Predict'):
    y_pred = loaded_model.predict(dataInput)
    st.success(f'Predict {y_pred[0]}')