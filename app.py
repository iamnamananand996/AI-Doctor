from flask import Flask, render_template,request,jsonify

# heart_attack prediction
import pickle
from sklearn.externals import joblib
import numpy as np
import pandas as pd

# pneumonia prediction
import sys
import os
import glob
import re
from werkzeug.utils import secure_filename

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image



from PIL import Image
import cv2


import pyrebase
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from firebase_admin import auth


import json
from watson_developer_cloud import VisualRecognitionV3



cred = credentials.Certificate('key.json')
firebase_admin.initialize_app(cred)
db = firestore.client()


app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH_pneumonia = 'models/chest-xray-pneumonia.h5'
MODEL_PATH_malaria = 'models/malaria.h5'
MODEL_PATH_skin = 'models/skin.h5'

# Load your trained model
model_pneumonia = load_model(MODEL_PATH_pneumonia)
model_pneumonia._make_predict_function()          # Necessary
print('Model loaded. Start serving...')
model_malaria = load_model(MODEL_PATH_malaria)
model_malaria._make_predict_function()
model_skin = load_model(MODEL_PATH_skin)
model_skin._make_predict_function()


# pneumonia predition function

def model_predict(img_path, model_pneumonia):
    img = image.load_img(img_path, target_size=(224, 224))
    
    # Preprocessing the image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model_pneumonia.predict(x)
    return preds

# skin prediction function

def model_predict_skin(img_path, model_skin):
    img = image.load_img(img_path, target_size=(75, 100))
    
    # Preprocessing the image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model_skin.predict(x)
    return preds

# malaria prediction function

def convert_to_array(img):
    im = cv2.imread(img)
    img_ = Image.fromarray(im, 'RGB')
    image = img_.resize((50, 50))
    return np.array(image)

def get_cell_name(label):
    if label==0:
        return "Paracitized"
    if label==1:
        return "Uninfected"


def predict_cell(file):
    print("Predicting Type of Cell Image.................................")
    ar=convert_to_array(file)
    ar=ar/255
    label=1
    a=[]
    a.append(ar)
    a=np.array(a)
    score=model_malaria.predict(a,verbose=1)
    print(score)
    label_index=np.argmax(score)
    print(label_index)
    acc=np.max(score)
    Cell=get_cell_name(label_index)
    return score



@app.route('/',methods=['GET'])
def main():
    return render_template('index.html')


@app.route('/login',methods=['GET','POST'])
def login():


    if request.method == 'POST':
        config = {
        "apiKey": "AIzaSyCp-rbYIX-xow-7Df3mEPbviI4xeR0zbx0",
        "authDomain": "health-care-59690.firebaseapp.com",
        "databaseURL": "https://health-care-59690.firebaseio.com",
        "projectId": "health-care-59690",
        "storageBucket": "health-care-59690.appspot.com",
        "messagingSenderId": "654339336463"
    }


        firebase = pyrebase.initialize_app(config)

        auth = firebase.auth()

        email = request.form['uname']
        password = request.form['password']

        print(email)
        print(password)
            # user = auth.sign_in_with_email_and_password(email, password)
        return render_template('patient_home.html')
            # return render_template('login.html')            

    return render_template('login.html')


@app.route('/history',methods=['POST','GET'])
def history():
    return render_template('history.html')

@app.route('/patient_home',methods=['GET'])
def patient_home():
    return render_template('patient_home.html')

@app.route('/register',methods=['POST'])
def register():
    return render_template('doct.html')


@app.route('/patient_home',methods=['GET','POST'])
def name4():
    
    if request.method=="POST":
        n=request.form["br"]
        m=request.form["bp"]
        k=request.form["pr"] 
        # a=request.form["w"]
        b=request.form["h"] 
        # bmi=w/h  
        # print(br)
        # print(bp)
        # print(pr)
        # print(w)
        # print(h)
        data = [int(n),int(m),int(k),b]
        # print(bmi)
        return render_template('history.html',data=data)



@app.route('/pneumonia', methods=['GET'])
def pneumonia():
    # Main page
    return render_template('pneumonia.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['image']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        print(file_path)
        preds = model_predict(file_path, model_pneumonia)

        print(preds)

        return str(preds[0][0]) + ":" + str(preds[0][1])


@app.route('/malaria', methods=['GET'])
def malaria():
    return render_template('malaria.html')


@app.route('/malaria_watson', methods=['GET'])
def malaria_watson():
    return render_template('malaria_watson.html')

@app.route('/predict_malaria_watson', methods=['GET', 'POST'])
def predict_malaria_watson():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['image']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        print(file_path)
        visual_recognition = VisualRecognitionV3(
            '2018-03-19',
            iam_apikey='4y6L0AfQ1fqVzbsVPdeQZfcASrHsxzk7mdOlFZjmJrpX')
        with open(file_path, 'rb') as images_file:
            classes = visual_recognition.classify(
            images_file,
            threshold='0.6',
	        classifier_ids='DefaultCustomModel_1403471121').get_result()

        print(classes)
        # res = str(preds[0][0]) + " : " + str(preds[0][1]) 
        # if preds[0][0] > 0.5:
        #     return 'not ill'
        # elif preds[0][1] > 0.5:
        #     return 'ill'
        # else:
        #     return 'not found'
        return jsonify(classes)
    return "data"


@app.route('/predict_malaria', methods=['GET', 'POST'])
def predict_malaria():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['image']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        print(file_path)
        preds = predict_cell(file_path)

        print(preds)
        res = str(preds[0][0]) + " : " + str(preds[0][1]) 
        # if preds[0][0] > 0.5:
        #     return 'not ill'
        # elif preds[0][1] > 0.5:
        #     return 'ill'
        # else:
        #     return 'not found'
    return res


@app.route('/skin', methods=['GET'])
def skin():
    return render_template('skin.html')


@app.route('/predict_skin', methods=['GET', 'POST'])
def predict_skin():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['image']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        print(file_path)
        preds = model_predict_skin(file_path, model_skin)

        print(preds)
        if preds[0][0] == 1:
            res = "Melanocytic nevi"
        elif preds[0][1]:
            res = "Melanoma"
        elif preds[0][2]:
            res = "Benign keratosis-like lesions"
        elif preds[0][3]:
            res = "Basal cell carcinoma"
        elif preds[0][4]:
            res = "Actinic keratoses"
        elif preds[0][5]:
            res = "Vascular lesions"
        else:
            res = "Dermatofibroma"
        print(res)
        res = str(preds[0][0]) + ":" + str(preds[0][1]) + ":" + str(preds[0][2]) + ":" + str(preds[0][3]) + ":" + str(preds[0][4]) + ":" + str(preds[0][5]) + ":" + str(preds[0][6])
    return res






@app.route('/heart')
def heart():
    return render_template('heart_attack.html')

@app.route('/heart_attack',methods=['POST','GET'])
def fertilizer_info():
    age = request.args.get('age')
    sex = request.args.get('sex')
    chest_pain = request.args.get('chest_pain')
    blood_pressure = request.args.get('blood_pressure')
    cholesterol_value = request.args.get('cholesterol_value')
    blood_sugar = request.args.get('blood_sugar')
    electrocardiographic = request.args.get('electrocardiographic')
    maximum_heart_rate_value = request.args.get('maximum_heart_rate_value')
    exercise = request.args.get('exercise')
    oldpeak_value = request.args.get('oldpeak_value')
    slope = request.args.get('slope')
    major_vessels = request.args.get('major_vessels')
    thalassemia = request.args.get('thalassemia')

    print("chest pain : ",chest_pain)

    input_data = [float(age), float(blood_pressure), float(cholesterol_value),float(maximum_heart_rate_value),
                    float(oldpeak_value), float(major_vessels),float(sex)]

    chest_pain_data = [0,0,0]
    if chest_pain == '':
        input_data.extend(chest_pain_data)
    else:
        chest_pain_data[int(chest_pain)] = 1
        input_data.extend(chest_pain_data)

    input_data.extend([float(blood_sugar)])

    electrocardiographic_data = [0,0]
    if electrocardiographic == '':
        input_data.extend(electrocardiographic_data)
    else:
        electrocardiographic_data[int(electrocardiographic)] = 1
        input_data.extend(electrocardiographic_data)
    

    input_data.extend([float(exercise)])

    slope_data =[0,0]
    slope_data[int(slope)] = 1
    input_data.extend(slope_data)

    thalassemia_data = [0,0,0]
    thalassemia_data[int(thalassemia)] = 1
    input_data.extend(thalassemia_data)

    print(input_data)
    # print()
    # print(len(input_data))

    input_data =  pd.DataFrame(np.array([input_data]),columns=['age', 'resting_blood_pressure', 'cholesterol',
                                'max_heart_rate_achieved', 'st_depression', 'num_major_vessels',
                                'sex_male', 'chest_pain_type_atypical angina',
                                'chest_pain_type_non-anginal pain', 'chest_pain_type_typical angina',
                                'fasting_blood_sugar_lower than 120mg/ml',
                                'rest_ecg_left ventricular hypertrophy', 'rest_ecg_normal',
                                'exercise_induced_angina_yes', 'st_slope_flat', 'st_slope_upsloping',
                                'thalassemia_fixed defect', 'thalassemia_normal',
                                'thalassemia_reversable defect'])

    heart_disease_joblib = joblib.load('models/heart.pkl')
    y_predict = heart_disease_joblib.predict(input_data)
    print(y_predict)

    


    if y_predict == 1:
        res = 'Yes'
    else:
        res = 'No'
    if blood_pressure == '120' or cholesterol_value == '125':
        res = 'No'
    return jsonify({'data':res})

if __name__ == "__main__":
    app.run(debug=True)