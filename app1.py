import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import cv2
import numpy as np
from keras.models import model_from_json
from streamlit_lottie import st_lottie
import matplotlib.pyplot as plt
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
#from decouple import config
# loading the saved models
# (Assuming your models are in the same directory as your Streamlit app)

diabetes_model = pickle.load(open("C:/Users/91934/OneDrive/Desktop/MPV2/predictions/diabetes_model.sav",'rb'))

heart_disease_model = pickle.load(open("C:/Users/91934/OneDrive/Desktop/MPV2/predictions/Heart_disease_model.sav",'rb'))

parkinsons_model = pickle.load(open("C:/Users/91934/OneDrive/Desktop/MPV2/predictions/parkinson_model.sav",'rb'))

liver_model = pickle.load(open("C:/Users/91934/OneDrive/Desktop/MPV2/predictions/Liver_disease_model.sav",'rb'))

Lung_cancer_model = pickle.load(open("C:/Users/91934/OneDrive/Desktop/MPV2/predictions/lung_cancer_model.sav",'rb'))

model=model_from_json(open('C:/Users/91934/OneDrive/Desktop/MPV2/predictions/tumor.json','r').read())

model.load_weights('C:/Users/91934/OneDrive/Desktop/MPV2/predictions/Tumor.h5')

kidney_model = pickle.load(open("C:/Users/91934/OneDrive/Desktop/MPV2/predictions/kidney_disease_model.sav",'rb'))

insurance_model = pickle.load(open("C:/Users/91934/OneDrive/Desktop/MPV2/predictions/insurence_predict_model.sav",'rb'))

def format_currency(amount, currency='INR'):
    
    formatted_amount = '{:,.2f}'.format(amount)
    
    if currency == 'INR':
        return '₹' + formatted_amount
    elif currency == 'USD':
        return '$' + formatted_amount
    return formatted_amount

# Function for Diabetes Prediction
def diabetes_prediction():
    st.header('Diabetes Prediction')

   # getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
        
    with col2:
        Glucose = st.text_input('Glucose Level')
    
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
    
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    
    with col2:
        Insulin = st.text_input('Insulin Level')
    
    with col3:
        BMI = st.text_input('BMI value')
    
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    
    with col2:
        Age = st.text_input('Age of the Person')
    
    
    # code for Prediction
    diab_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        diab_prediction = diabetes_model.predict([[int(Pregnancies), int(Glucose), int(BloodPressure), int(SkinThickness), int(Insulin), float(BMI), float(DiabetesPedigreeFunction), int(Age)]])
        
        if (diab_prediction[0] == 1):
          diab_diagnosis = 'The person is diabetic'
        else:
          diab_diagnosis = 'The person is not diabetic'
        
    st.success(diab_diagnosis)

# Function for Heart Disease Prediction
def heart_disease_prediction():
    st.header('Heart Disease Prediction')

    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Age')
        
    with col2:
        sex = st.text_input('Sex')
        
    with col3:
        cp = st.text_input('Chest Pain types')
        
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
        
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')
        
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
        
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')
        
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')
        
    with col3:
        exang = st.text_input('Exercise Induced Angina')
        
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')
        
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')
        
    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')
        
    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
        
        
     
     
    # code for Prediction
    heart_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Heart Disease Test Result'):
        heart_prediction = heart_disease_model.predict([[int(age), int(sex), int(cp), int(trestbps), int(chol), int(fbs), int(restecg),int(thalach),int(exang),float(oldpeak),int(slope),int(ca),int(thal)]])                          
        
        if (heart_prediction[0] == 1):
          heart_diagnosis = 'The person is having heart disease'
        else:
          heart_diagnosis = 'The person does not have any heart disease'
        
    st.success(heart_diagnosis)

# Function for Parkinson's Prediction
def parkinsons_prediction():
    st.header("Parkinson's Disease Prediction")

    col1, col2, col3, col4, col5 = st.columns(5)  
    
    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')
        
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')
        
    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')
        
    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')
        
    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
        
    with col1:
        RAP = st.text_input('MDVP:RAP')
        
    with col2:
        PPQ = st.text_input('MDVP:PPQ')
        
    with col3:
        DDP = st.text_input('Jitter:DDP')
        
    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')
        
    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
        
    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')
        
    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')
        
    with col3:
        APQ = st.text_input('MDVP:APQ')
        
    with col4:
        DDA = st.text_input('Shimmer:DDA')
        
    with col5:
        NHR = st.text_input('NHR')
        
    with col1:
        HNR = st.text_input('HNR')
        
    with col2:
        RPDE = st.text_input('RPDE')
        
    with col3:
        DFA = st.text_input('DFA')
        
    with col4:
        spread1 = st.text_input('spread1')
        
    with col5:
        spread2 = st.text_input('spread2')
        
    with col1:
        D2 = st.text_input('D2')
        
    with col2:
        PPE = st.text_input('PPE')
        
    
    
    # code for Prediction
    parkinsons_diagnosis = ''
    
    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):
        parkinsons_prediction = parkinsons_model.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ,DDP,Shimmer,Shimmer_dB,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]])                          
        
        if (parkinsons_prediction[0] == 1):
          parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
          parkinsons_diagnosis = "The person does not have Parkinson's disease"
        
    st.success(parkinsons_diagnosis)

# Function for Liver Disease Prediction
def liver_disease_prediction():
    st.header('Liver Disease Prediction')

    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        Age = st.text_input('Age')
    with col2:
        Gender = st.text_input('Gender')
    with col3:
       Total_Bilirubin = st.text_input('Total_Bilirubin') 
    with col4:
        Direct_Bilirubin = st.text_input('Direct_Bilirubin')
    with col5:
        Alkaline_Phosphotase = st.text_input('Alkaline_Phosphotase')
    with col1:
        Alamine_Aminotransferase = st.text_input('Alamine_Aminotransferase')
    with col2:
        Aspartate_Aminotransferase = st.text_input('Aspartate_Aminotransferase')
    with col3:
        Total_Protiens = st.text_input('Total_Protiens')
    with col4:
        Albumin = st.text_input('Albumin')
    with col5:
        Albumin_and_Globulin_Ratio = st.text_input('Albumin_and_Globulin_Ratio')
        
    
    liver_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Liver Disease Test Result'):
        liver_prediction = liver_model.predict([[int(Age),int(Gender), float(Total_Bilirubin), float(Direct_Bilirubin), int(Alkaline_Phosphotase), int(Alamine_Aminotransferase), int(Aspartate_Aminotransferase),float(Total_Protiens),float(Albumin),float(Albumin_and_Globulin_Ratio)]])                      
        
        if (liver_prediction[0] == 1):
          liver_diagnosis = 'The person is having Liver disease'
        else:
          liver_diagnosis = 'The person does not have any Liver disease'
        
    st.success(liver_diagnosis)
    
# Function for Lung Cancer
def lung_cancer_prediction():
    st.header('Lung Cancer Prediction')
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        GENDER = st.number_input("GENDER")

    with col2:
        SMOKING = st.number_input("SMOKING")
    
    with col3:
        YELLOW_FINGERS = st.number_input("YELLOW_FINGERS")
    
    with col4:
        ANXIETY = st.number_input("ANXIETY")
    
    with col1:
        PEER_PRESSURE = st.number_input("PEER_PRESSURE")
    
    with col2:
        CHRONIC_DISEASE = st.number_input("CHRONIC DISEASE")
    
    with col3:
        FATIGUE = st.number_input("FATIGUE")
    
    with col4:
        ALLERGY = st.number_input("ALLERGY")
    
    with col1:
        WHEEZING = st.number_input("WHEEZING")
    
    with col2:
        ALCOHOL_CONSUMING = st.number_input("ALCOHOL CONSUMING")
    
    with col3:
        COUGHING = st.number_input("COUGHING")
    
    with col4:
        SHORTNESS_OF_BREATH = st.number_input("SHORTNESS OF BREATH")
    
    with col1:
        SWALLOWING_DIFFICULTY = st.number_input("SWALLOWING DIFFICULTY")
    
    with col2:
        CHEST_PAIN = st.number_input("CHEST PAIN")
        
    lung_cancer_result = " "
    
    # creating a button for Prediction
    
    if st.button("Lung Cancer Test Result"):
        lung_cancer_report = Lung_cancer_model.predict([[GENDER, SMOKING, YELLOW_FINGERS, ANXIETY, PEER_PRESSURE, CHRONIC_DISEASE, FATIGUE, ALLERGY, WHEEZING, ALCOHOL_CONSUMING, COUGHING, SHORTNESS_OF_BREATH, SWALLOWING_DIFFICULTY, CHEST_PAIN]])
        
        if (lung_cancer_report[0] == 0):
          lung_cancer_result = "Hurrah! You have no Lung Cancer."
        else:
          lung_cancer_result = "Sorry! You have Lung Cancer."
        
    st.success(lung_cancer_result)
# Function for Kidney Disease Prediction
def Kidney_Disease_Prediction():
    st.title("Kidney Disease Prediction")

    col1, col2, col3, col4 , col5= st.columns(5)

    with col1:
        id = st.number_input("ID")

    with col2:
        age = st.number_input("AGE")

    with col3:
        bp = st.number_input("Blood Pressure")

    with col4:
        sg = st.number_input("Serum Glucose")

    with col5:
        al = st.number_input("Albumin")

    with col1:
        su = st.number_input("sodium")

    with col2:
        rbc = st.number_input("Red Blood Cells")

    with col3:
        pc = st.number_input("Pus Cells")

    with col4:
        pcc = st.number_input("Pus Cell Clumps")
        
    with col5:
        ba = st.number_input("Bacteria")

    with col1:
        bgr = st.number_input("Blood Glucose Random")

    with col2:
        bu = st.number_input("Bilirubin")

    with col3:
        sc = st.number_input("Serum Creatinine")

    with col4:
        sod = st.number_input("Sodium")

    with col5:
        pot = st.number_input("Potassium")

    with col1:
        hemo = st.number_input("Hemoglobin")

    with col2:
        pcv = st.number_input("Packed Cell Volume")

    with col3:
        wc = st.number_input("White Blood Cell Count")
        
    with col4:
        rc = st.number_input("Red Blood Cell Count")

    with col5:
        htn = st.number_input("Hypertension")
        
    with col1:
        dm = st.number_input("Diabetes mellitus")

    with col2:
        cad = st.number_input("Coronary Artery Disease")

    with col3:
        appet = st.number_input("Appetite")
        
    with col4:
        pe = st.number_input("Pedal Edema")

    with col5:
        ane = st.number_input("Anemia")
        
    kidney_disease_result = " "
    
    # creating a button for Prediction
    
    if st.button("Kidney Disease Test Result"):
        kidney_disease_report = kidney_model.predict([[id, age, bp, sg, al, su, rbc, pc, pcc, ba, bgr, bu, sc, sod, pot, hemo, pcv, wc, rc, htn, dm, cad, appet, pe, ane]])
        
        if (kidney_disease_report[0] == 1):
          kidney_disease_result = "Hurrah! You have no Kidney Disease."
        else:
          kidney_disease_result = "Sorry! You have Kidney Disease."
        
    st.success(kidney_disease_result)
def insurance_charges_prediction(model):

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age")

    with col2:
        sex = st.number_input("Sex")

    with col3:
        bmi = st.number_input("BMI")

    with col1:
        children = st.number_input("Children")

    with col2:
        smoker = st.number_input("Smoker")

    with col3:
        region = st.number_input("Region")

    insurance_charges_result = " "

    if st.button("Insurance Charges Test Result"):
        insurance_charges_report = insurance_model.predict([[age, sex, bmi, children, smoker, region]])
        formatted_charges = format_currency(insurance_charges_report[0], currency='INR')
        insurance_charges_result = f"Your predicted insurance charges: {formatted_charges}"
    st.success(insurance_charges_result)
# Function for Brain Tumor Detection
def brain_tumor_detection(model):
    st.title('Brain Tumor Detection')

    # File upload for brain tumor detection
    uploaded_file = st.file_uploader("Upload an image for brain tumor detection", type=['jpeg', 'png', 'jpg', 'webp'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button("Detect Brain Tumor"):
            result, opencvImage = detect(image=image)
            predictions = model.predict(result.reshape(1, 100, 100, 1))
            st.image(opencvImage, caption=f'Tumor Detection: {predictions[0][0]}', use_column_width=True)
            st.success(f'Tumor Detection: {predictions[0][0]}')

# Function to preprocess the image for brain tumor detection
def detect(image):
    opencvImage = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    grey = cv2.cvtColor(opencvImage, cv2.COLOR_BGR2GRAY)
    grey = cv2.resize(grey, (100, 100))
    return grey, opencvImage

st.set_page_config(
    page_title="OmniHealth Explorer",
    page_icon=":hospital:",
    layout="centered",
)
def create_bar_graph(data, labels, colors, legend_labels, title):
    fig, ax = plt.subplots()
    bars = plt.bar(labels, data, color=colors)

    # Add legend with labels
    ax.legend(bars, legend_labels)
    fig.set_size_inches(6, 3)

    st.subheader(title)
    st.pyplot(fig)

with st.sidebar:
    st.title('Menu Area')

    # Main menu options
    selected = st.selectbox('', ['About Us','Home', 'Predictions', 'Insurances Charges', 'Contact Us'])
    
if selected == 'About Us':
    st.markdown(
    """
    <style>
    .about-title {
        background-color: #000000;
        padding: 7px; border-radius: 34px;
        display: inline-block; color: white;
        text-shadow: 2px 2px 4px #1e90ff;
        font-size: 42px; text-align: center;
        width: 104%; margin: 2px auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
    st.markdown(
    '<div class="about-title">About Us</div>',
    unsafe_allow_html=True
)

    css = """
    <style>
    /* CSS for the typing animation */
    @keyframes typing {
        0% { width: 52%; }
        50% { width: 104%; }
        100% { width: 104%; }
    }

    /* CSS for the text box */
    .text-box {
        background-color: white;
        padding: 16px; border-radius: 69px;
        display: flex; align-items: center;
        justify-content: center; 
        overflow: hidden; text-align: center; 
        width: 60%; margin: 18px auto;
    }

    /* CSS for the text inside the box */
    .typing-text {
        white-space: nowrap; overflow: hidden; 
        animation: typing 5s linear infinite; 
        font-size: 24px; font-weight: bold; 
        color: black;
    }
    
    /* CSS for the formatted text box */
    .formatted-text-box {
        background-color: white; 
        padding: 20px; border-radius: 10px; 
        color: black; width: 100%; 
        display: flex; justify-content: center; 
        align-items: center;
    }

    /* CSS for the text inside the formatted box */
    .text {
        text-align: justify; 
    }
    </style>
"""

# Displaying the CSS
    st.markdown(css, unsafe_allow_html=True)

# Displaying the "Welcome to Omni Health Explorer" box
    st.markdown('<div class="text-box"><div class="typing-text">Welcome to Omni Health Explorer.</div></div>', unsafe_allow_html=True)

# Displaying the formatted text box
    st.markdown(
    """
    <div class="formatted-text-box">
        <div class="text">
            <ul>
                <li>In this website we have created a user-friendly interface by which the user can get accurate results based on the input provided by them.</li>
                <li>This website is divided into five parts: Home, About Us, Contact Us, Predictions, and Insurance Charges.</li>
                <li>Home Page consists of brief information about the diseases predicted on this website.</li>
                <li>Predictions Page consists of all the prediction models where the user can get predictions based on the input provided by them.</li>
                <li>Insurance Charges Page consists of the prediction model for the Insurance Charges.</li>
                <li>Contact Us consists of the contact details of the website.</li>
            </ul>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

    

# Home Section
elif selected == 'Home':
    
    #st.title('Explore Health Avenues')
    st.markdown(
    """
    <style>
    .title-box {
        border: 2px solid #000000; 
        padding: 9px; 
        border-radius: 59px;
        background-color: #000000; 
        animation: pop-up 1s ease-in-out forwards;
        text-align: center;
        font-size: 36px;
    }
    
    @keyframes pop-up {
        0% { transform: scale(0.5); opacity: 0; }
        100% { transform: scale(1); opacity: 1; }
    }

    .pop-up-words {
        display: inline-block;
        animation: pop-up-word 1s ease-in-out forwards; 
        opacity: 0;
        color: white;
        text-shadow: 2px 2px 4px #87CEFA;
    }

    @keyframes pop-up-word {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }

    </style>
    """,
    unsafe_allow_html=True
)

# Title
    st.markdown(
    '<div class="title-box"><span class="pop-up-words">HOME</span></div>',
    unsafe_allow_html=True
)
    
    # Content
    st.subheader('Explore the power of predictions with our disease detection system!')

    # Overview
    st.write(
        "Our system leverages machine learning algorithms to predict various health conditions. "
        "Explore the detailed descriptions of the diseases covered:"
    )

    # Diabetes Prediction
    
    st.markdown("""
    <div style="
        display: flex; flex-direction: column;
        justify-content: center; align-items: center;
        width: fit-content; padding: 20px;
        background-color: white; color: black;
        border-radius: 10px;
        ">
        <div style="margin-bottom: 20px;">
            <span style="font-size: 16px;"><strong>1. Diabetes Prediction:</strong></span>
        </div>
        <div style="width: fit-content;">
            <p style="font-family: Arial, sans-serif; font-size: 16px; text-align: justify;"> 
                Diabetes prediction is a crucial task in healthcare. Machine learning models can be used to predict diabetes status from features such as glucose, blood pressure, skin thickness, insulin, and BMI. The Pima Indians Diabetes Database is a famous dataset used for this purpose. The data analysis part is done in a data science life cycle. Exploratory data analysis (EDA) is one of the most important steps in the data science project life cycle. Here one will need to know how to make inferences from the visualizations and data analysis. Model building is another important step. Here we will be using 4 ML models and then we will choose the best performing model. Key factors include the number of pregnancies, glucose level, blood pressure, skin thickness, insulin level, BMI, diabetes pedigree function, and age.
            </p>
        </div>
        <div style="margin-top: 20px;">
            <img src="https://julianhealthcare.com/wp-content/uploads/2019/07/Diabetes.jpg" alt="Diabetes Prediction" style="width: 400px; display: block; margin: 0 auto;">
        </div>
        <div style="margin-top: 20px;">
            <strong>Dataset:</strong> 
            <a href="https://www.kaggle.com/uciml/pima-indians-diabetes-database" target="_blank" rel="noopener noreferrer">Diabetes Dataset</a>
        </div>
    </div>
""", unsafe_allow_html=True)

    create_bar_graph(
        data=np.random.randint(50, 200, 3),
        labels=['Feature 1', 'Feature 2', 'Feature 3'],
        colors=['blue', 'green', 'orange'],
        legend_labels=['Label 1', 'Label 2', 'Label 3'],
        title="Bar Graph:"
    )
    
    # Heart Disease Prediction
    st.markdown("""
    <div style="
        display: flex; flex-direction: column;
        justify-content: center; align-items: center;
        width: fit-content; padding: 20px;
        background-color: white; color: black;
        border-radius: 10px;
        ">
        <div style="margin-bottom: 20px;">
            <span style="font-size: 16px;"><strong>2. Heart Disease Prediction:</strong></span>
        </div>
        <div style="width: fit-content;">
            <p style="font-family: Arial, sans-serif; font-size: 16px; text-align: justify;"> 
                Heart Disease Prediction evaluates cardiovascular health by predicting the likelihood of heart disease. Parameters considered include age, sex, chest pain types, resting blood pressure, serum cholesterol level, fasting blood sugar, resting electrocardiographic results, maximum heart rate achieved, exercise-induced angina, ST depression induced by exercise, slope of the peak exercise ST segment, major vessels colored by fluoroscopy, and thal.
            </p>
        </div>
        <div style="margin-top: 20px;">
            <img src="https://d2icp22po6iej.cloudfront.net/wp-content/uploads/2018/08/PD-AND-THE-HEART3.jpeg" alt="Heart Disease Prediction" style="width: 400px; display: block; margin: 0 auto;">
        </div>
        <div style="margin-top: 20px;">
            <strong>Dataset:</strong> 
            <a href="https://www.kaggle.com/code/georgyzubkov/heart-disease-exploratory-data-analysis" target="_blank" rel="noopener noreferrer">Heart Disease Dataset</a>
        </div>
    </div>
""", unsafe_allow_html=True)

    create_bar_graph(
        data=np.random.randint(50, 200, 3),
        labels=['Feature 1', 'Feature 2', 'Feature 3'],
        colors=['blue', 'green', 'orange'],
        legend_labels=['Label 1', 'Label 2', 'Label 3'],
        title="Bar Graph:"
    )
    
    # Parkinson's Disease Prediction
    
    st.markdown("""
    <div style="
        display: flex; flex-direction: column;
        justify-content: center; align-items: center;
        width: fit-content; padding: 20px;
        background-color: white; color: black;
        border-radius: 10px;
        ">
        <div style="margin-bottom: 20px;">
            <span style="font-size: 16px;"><strong>3. Parkinson's Disease Prediction:</strong></span>
        </div>
        <div style="width: fit-content;">
            <p style="font-family: Arial, sans-serif; font-size: 16px; text-align: justify;"> 
                Our Parkinson's Disease Prediction model utilizes voice and speech features such as pitch, jitter, and shimmer to assess the likelihood of Parkinson's disease. By analyzing these characteristics, the system helps in early detection and monitoring of Parkinson's disease, facilitating timely medical support.
            </p>
        </div>
        <div style="margin-top: 20px;">
            <img src="https://atlphysio.com/wp-content/uploads/2021/12/parkinsons-disease.jpg" alt="Parkinson's Disease Prediction" style="width: 400px; display: block; margin: 0 auto;">
        </div>
        <div style="margin-top: 20px;">
            <strong>Dataset:</strong> 
            <a href="https://www.kaggle.com/nidaguler/parkinsons-data-set" target="_blank" rel="noopener noreferrer">Parkinson's Disease Dataset</a>
        </div>
    </div>
""", unsafe_allow_html=True)


    create_bar_graph(
    data=np.random.randint(50, 200, 5),
    labels=['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5'],
    colors=['blue', 'green', 'orange'],
    legend_labels=['Label 1 & 2', 'Label 3 & 4', 'Label 5'],
    title="Bar Graph:"
    )
    
    # Liver Disease Prediction
    
    st.markdown("""
    <div style="
        display: flex; flex-direction: column;
        justify-content: center; align-items: center;
        width: fit-content; padding: 20px;
        background-color: white; color: black;
        border-radius: 10px;
        ">
        <div style="margin-bottom: 20px;">
            <span style="font-size: 16px;"><strong>4. Liver Disease Prediction:</strong></span>
        </div>
        <div style="width: fit-content;">
            <p style="font-family: Arial, sans-serif; font-size: 16px; text-align: justify;"> 
                The Liver Disease Prediction model evaluates liver health by considering factors like age, gender, bilirubin levels, enzyme levels, and protein concentrations. By analyzing these biochemical markers, the system predicts the likelihood of liver disease, aiding in early identification and personalized health management. Key factors include age, gender, total bilirubin, direct bilirubin, alkaline phosphatase, alanine aminotransferase, aspartate aminotransferase, total proteins, albumin, and albumin-to-globulin ratio.
            </p>
        </div>
        <div style="margin-top: 20px;">
            <img src="https://www.drcarolyndean.net/wp-content/uploads/2020/01/Liver-Disease.jpg" alt="Liver Disease Prediction" style="width: 400px; display: block; margin: 0 auto;">
        </div>
        <div style="margin-top: 20px;">
            <strong>Dataset:</strong> 
            <a href="https://www.kaggle.com/jeevannagaraj/indian-liver-patient-dataset" target="_blank" rel="noopener noreferrer">Liver Disease Dataset</a>
        </div>
    </div>
""", unsafe_allow_html=True)

    
    create_bar_graph(
        data=np.random.randint(50, 200, 5),
        labels=['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5'],
        colors=['blue', 'green', 'orange'],
        legend_labels=['Label 1 & 2', 'Label 3 & 4', 'Label 5'],
        title="Bar Graph:"
    )
    
    # Brain Tumor Detection
    
    st.markdown("""
    <div style="
        display: flex; flex-direction: column;
        justify-content: center; align-items: center;
        width: fit-content; padding: 20px;
        background-color: white; color: black;
        border-radius: 10px;
        ">
        <div style="margin-bottom: 20px;">
            <span style="font-size: 16px;"><strong>5. Brain Tumor Detection:</strong></span>
        </div>
        <div style="width: fit-content;">
            <p style="font-family: Arial, sans-serif; font-size: 16px; text-align: justify;"> 
                The Brain Tumor Detection system employs advanced convolutional neural networks (CNNs) to analyze MRI scans. By detecting abnormalities and patterns indicative of brain tumors, the system provides invaluable insights for clinicians, assisting in the timely diagnosis and treatment of brain-related conditions.
            </p>
        </div>
        <div style="margin-top: 20px;">
            <img src="https://media.sciencephoto.com/image/c0370760/800wm/C0370760-Astrocytoma_brain_cancer,_MRI_scan.jpg" alt="Brain Tumor Detection" style="width: 400px; display: block; margin: 0 auto;">
        </div>
        <div style="margin-top: 20px;">
            <strong>Dataset:</strong> 
            <a href="https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection" target="_blank" rel="noopener noreferrer">Brain Tumor Dataset</a>
        </div>
    </div>
""", unsafe_allow_html=True)

    create_bar_graph(
        data=np.random.randint(50, 200, 5),
        labels=['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5'],
        colors=['blue', 'green', 'orange'],
        legend_labels=['Label 1 & 2', 'Label 3 & 4', 'Label 5'],
        title="Bar Graph:"
    )
    
    # Lung Cancer Prediction
    
    st.markdown("""
    <div style="
        display: flex; flex-direction: column;
        justify-content: center; align-items: center;
        width: fit-content; padding: 20px;
        background-color: white; color: black;
        border-radius: 10px;
        ">
        <div style="margin-bottom: 20px;">
            <span style="font-size: 16px;"><strong>6. Lung Cancer Prediction:</strong></span>
        </div>
        <div style="width: fit-content;">
            <p style="font-family: Arial, sans-serif; font-size: 16px; text-align: justify;"> 
                Lung cancer is a malignant disease characterized by the uncontrolled growth of cells in the lungs, typically forming tumors. It encompasses two primary types: non-small cell lung cancer (NSCLC) and small cell lung cancer (SCLC), with NSCLC being more prevalent. Major risk factors include smoking, exposure to secondhand smoke, radon gas, asbestos, family history, and certain lung diseases. Symptoms often include a persistent cough, coughing up blood, chest pain, shortness of breath, hoarseness, weight loss, fatigue, and recurring respiratory infections. Diagnosis involves imaging tests like X-rays and CT scans, along with biopsies for confirmation. Treatment options vary based on cancer type and stage, including surgery, chemotherapy, radiation therapy, targeted therapy, and immunotherapy, with prognosis influenced by factors like stage at diagnosis and overall health. Early detection and intervention are crucial for improving outcomes, although lung cancer is frequently diagnosed at advanced stages, presenting challenges in treatment.
            </p>
        </div>
        <div style="margin-top: 20px;">
            <img src="https://th.bing.com/th/id/R.c02d70b57be42814073949603f51e78d?rik=eqdItqpiXUGj3A&riu=http%3a%2f%2fgenassistabcs.com%2fwp-content%2fuploads%2f2016%2f12%2fCancer-Lung.jpeg&ehk=yp0JqXcE4BTXm4jzQvcVH1nrJ1Ye2wSZ5FJaHSZRc2w%3d&risl=&pid=ImgRaw&r=0" alt="Lung Cancer Prediction" style="width: 400px;">
        </div>
        <div style="margin-top: 20px;">
            <strong>Dataset:</strong> 
            <a href="https://www.kaggle.com/code/hasibalmuzdadid/lung-cancer-analysis-accuracy-96-4" target="_blank" rel="noopener noreferrer">Lung Cancer Dataset</a>
        </div>
    </div>
""", unsafe_allow_html=True)

    
    create_bar_graph(
        data=np.random.randint(50, 200, 4),
        labels=['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4'],
        colors=['blue', 'green'],
        legend_labels=['Label 1 & 2', 'Label 3 & 4',],
        title="Bar Graph:"
    )
    
    
    # Kidney Disease Prediction
    st.markdown("""
    <div style="
        display: flex; flex-direction: column;
        justify-content: center; align-items: center;
        width: fit-content; padding: 20px;
        background-color: white; color: black;
        border-radius: 10px;
        ">
        <div style="margin-bottom: 20px;">
            <span style="font-size: 16px;"><strong>7. Kidney Disease Prediction:</strong></span>
        </div>
        <div style="width: fit-content;">
            <p style="font-family: Arial, sans-serif; font-size: 16px; text-align: justify;"> 
                Kidney disease, also known as renal disease, encompasses a range of conditions that impair kidney function, leading to a buildup of toxins and waste in the body. These conditions can include acute kidney injury, chronic kidney disease (CKD), kidney stones, and various infections. Symptoms may include changes in urination patterns, fatigue, swelling, and difficulty concentrating. Causes of kidney disease vary from infections and autoimmune disorders to diabetes and high blood pressure. Diagnosis typically involves blood tests, urine tests, imaging studies, and kidney biopsies. Treatment options depend on the specific condition but may include medication, lifestyle changes, dialysis, or kidney transplantation. Early detection and management are crucial in preventing complications and preserving kidney function.
            </p>
        </div>
        <div style="margin-top: 20px;">
            <img src="https://th.bing.com/th/id/OIP.qglClfLrUNREkAuOUhu8XAHaE8?rs=1&pid=ImgDetMain" alt="Kidney Disease Prediction" style="width: 400px;">
        </div>
        <div style="margin-top: 20px;">
            <strong>Dataset:</strong> 
            <a href="https://www.kaggle.com/mahmoudlimam/preprocessed-chronic-kidney-disease-dataset" target="_blank" rel="noopener noreferrer">Kidney Disease Dataset</a>
        </div>
    </div>
""", unsafe_allow_html=True)

    
    create_bar_graph(
        data=np.random.randint(50, 200, 4),
        labels=['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4'],
        colors=['blue', 'green', 'orange'],
        legend_labels=['Label 1 & 2', 'Label 3 & 4', 'Label 5'],
        title="Bar Graph:"
    )
    st.info("Click on 'Predictions' in the sidebar to start exploring disease predictions.")
    
# Predictions Section
elif selected == 'Predictions':
    
    st.markdown("""
    <div style="
        background-color: white; border-radius: 16px; 
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1); 
        padding: -1px; text-align: center; 
        font-family: Arial, sans-serif; color: #333; 
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5); animation: popUp 0.5s ease-out;
        ">
        <h2 style='color: black;'><strong>Disease Predictions Overview</strong></h2>
    </div>

    <style>
        @keyframes popUp {
            0% {
                transform: scale(0);
            }
            100% {
                transform: scale(1);
            }
        }
    </style>
""", unsafe_allow_html=True)
    
    # Expander for Diabetes Prediction
    with st.expander("Diabetes Prediction"):
        st.write("Click to view Diabetes Prediction details.")
        diabetes_expander = st.checkbox('View Diabetes Prediction')
        if diabetes_expander:
            diabetes_prediction()

    # Expander for Heart Disease Prediction
    with st.expander("Heart Disease Prediction"):
        st.write("Click to view Heart Disease Prediction details.")
        heart_disease_expander = st.checkbox('View Heart Disease Prediction')
        if heart_disease_expander:
            heart_disease_prediction()

    # Expander for Parkinson's Disease Prediction
    with st.expander("Parkinson's Disease Prediction"):
        st.write("Click to view Parkinson's Disease Prediction details.")
        parkinsons_expander = st.checkbox('View Parkinson\'s Disease Prediction')
        if parkinsons_expander:
            parkinsons_prediction()

    # Expander for Liver Disease Prediction
    with st.expander("Liver Disease Prediction"):
        st.write("Click to view Liver Disease Prediction details.")
        liver_expander = st.checkbox('View Liver Disease Prediction')
        if liver_expander:
            liver_disease_prediction()

    # Expander for Brain Tumor Detection
    with st.expander("Brain Tumor Detection"):
        st.write("Click to view Brain Tumor Detection details.")
        brain_tumor_expander = st.checkbox('View Brain Tumor Detection')
        if brain_tumor_expander:
            brain_tumor_detection(model)
            
    # Expander for Lung Cancer Prediction
    with st.expander("Lung Cancer Prediction"):
        st.write("Click to view Lung Cancer Prediction details.")
        lung_expander = st.checkbox('View Lung Cancer Prediction')
        if lung_expander:
            lung_cancer_prediction()

    with st.expander("Kidney Disease Prediction"):
        st.write("Click to view Kidney Disease Prediction details.")
        Kidney_expander = st.checkbox('View Kidney Disease Prediction')
        if Kidney_expander:
            Kidney_Disease_Prediction()
            

elif selected == 'Insurances Charges':
    
    st.markdown("""
    <div style="
        text-align: center;
        ">
        <div style="
            display: inline-block;
            background-color: #f0f0f0;
            border-radius: 28px;
            padding: 12px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
            ">
            <h2 style='color: black; font-family: Arial, sans-serif; animation: moveHorizontal 3s infinite alternate;'><strong>Personalized Pricing for Your Coverage!</strong></h2>
        </div>
    </div>

    <style>
        @keyframes moveHorizontal {
            from {
                transform: translateX(-10px);
            }
            to {
                transform: translateX(10px);
            }
        }
    </style>
""", unsafe_allow_html=True)

    st.markdown("""
    <div style="
        text-align: center;
        animation: popUp 0.5s ease-out;
        ">
        <div style="
            display: inline-block;
            background-color: #ffffff;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
            color: black;
            margin: 18px auto;
            ">
            <h3 style='font-family: Arial, sans-serif; font-weight: bold; color: #333;'>Input Information</h3>
            <hr>
            <p style="text-align: left;"><strong>For gender:</strong><br>
            if You're male press 1, if female press 0. 
            </p>
            <p style="text-align: left;"><strong>if You Smoke then Press 1, or Press 0 if You don't smoke.</strong> 
            </p>
            <p style="text-align: left;"><strong>For Regions:</strong><br>
            1) If You are From South-East Region press 0<br>
            2) If Your are From South-West Region Press 1<br>
            3) If Your are From North-East Region Press 2<br>
            4) If Your are From North-West Region Press 3
            </p>
        </div>
    </div>

    <style>
        @keyframes popUp {
            0% {
                transform: scale(0);
            }
            100% {
                transform: scale(1);
            }
        }
    </style>
""", unsafe_allow_html=True)

    
    with st.expander("Insurance Charges Prediction",expanded=True):
        st.markdown("<h2 style='text-align: center; color: white; font-family: Arial, sans-serif;'>🚀 <strong>Insurance Charges Prediction</strong> 💰</h2>", unsafe_allow_html=True)
        st.write("Click to view Insurance Charges Prediction details.")
        insurance_expander = st.checkbox('View Insurance Charges Prediction')
        if insurance_expander:
            insurance_charges_prediction(model)


# Contact Us Section
elif selected == 'Contact Us':
    st.markdown("""
    <div style="
        background-color: white; 
        border-radius: 16px; 
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1); 
        padding: -1px; 
        text-align: center; 
        font-family: Arial, sans-serif; 
        color: #333; 
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
        animation: popUp 0.5s ease-out;
        ">
        <h2 style='color: black;'><strong>Contact Us!</strong></h2>
    </div>

    <style>
        @keyframes popUp {
            0% {
                transform: scale(0);
            }
            100% {
                transform: scale(1);
            }
        }
    </style>
""", unsafe_allow_html=True)
    
    name = st.text_input("Your Name")
    email = st.text_input("Your Email")
    query = st.text_area("Your Query", height=100)

    if st.button("Submit"):
        # Add code to handle the form submission, e.g., sending an email or saving to a database
        st.success("Your query has been submitted. We will get back to you soon.")

    # Add any additional contact information or details as needed
    st.write("For urgent matters, please contact us at saieshwarpennada1234@gmail.com.")

def set_bg_from_url(url, opacity=1):
    
    footer = """
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.0/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-gH2yIJqKdNHPEq0n4Mqa/HGKIhSkIHeL5AyhkYV8i59U5AR6csBvApHHNl/vI1Bx" crossorigin="anonymous">
    <footer>
        <div style='visibility: visible;margin-top:7rem;justify-content:center;display:flex;'>
            <p style="font-size:1.1rem;">
                &nbsp;
                <a href="https://www.linkedin.com/in/saieshwar-pennada-5281b123a/">
                    <svg xmlns="http://www.w3.org/2000/svg" width="23" height="23" fill="white" class="bi bi-linkedin" viewBox="0 0 16 16">
                        <path d="M0 1.146C0 .513.526 0 1.175 0h13.65C15.474 0 16 .513 16 1.146v13.708c0 .633-.526 1.146-1.175 1.146H1.175C.526 16 0 15.487 0 14.854V1.146zm4.943 12.248V6.169H2.542v7.225h2.401zm-1.2-8.212c.837 0 1.358-.554 1.358-1.248-.015-.709-.52-1.248-1.342-1.248-.822 0-1.359.54-1.359 1.248 0 .694.521 1.248 1.327 1.248h.016zm4.908 8.212V9.359c0-.216.016-.432.08-.586.173-.431.568-.878 1.232-.878.869 0 1.216.662 1.216 1.634v3.865h2.401V9.25c0-2.22-1.184-3.252-2.764-3.252-1.274 0-1.845.7-2.165 1.193v.025h-.016a5.54 5.54 0 0 1 .016-.025V6.169h-2.4c.03.678 0 7.225 0 7.225h2.4z"/>
                    </svg>          
                </a>
                &nbsp;
                <a href="https://github.com/SaiEshwar-Pennada">
                    <svg xmlns="http://www.w3.org/2000/svg" width="23" height="23" fill="white" class="bi bi-github" viewBox="0 0 16 16">
                        <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/>
                    </svg>
                </a>
            </p>
        </div>
    </footer>
"""
    st.markdown(footer, unsafe_allow_html=True)
    
    
    # Set background image using HTML and CSS
    st.markdown(
        f"""
        <style>
            body {{
                background: url('{url}') no-repeat center center fixed;
                background-size: cover;
                opacity: {opacity};
            }}
        </style>
        """,
        unsafe_allow_html=True
    )
# Set background image from URL
set_bg_from_url("https://images.everydayhealth.com/homepage/health-topics-2.jpg?w=768", opacity=0.875)
