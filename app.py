from flask import Flask,render_template,request
import requests
import pickle
import numpy as np
import sklearn
import pandas as pd
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/',methods = ['GET'])
def Home():
    return render_template('index.html')


standard_to = StandardScaler()
@app.route('/predict',methods = ['POST'])
def predict():
    if request.method == 'POST':
       fixed_acidity = float(request.form['fixed acidity'])
       volatile_acidity = float(request.form['volatile acidity'])
       citric_acid = float(request.form['citric acid'])
       residual_sugar = float(request.form['residual sugar'])
       chlorides = float(request.form['chlorides'])
       free_sulfur_dioxide = float(request.form['free sulfur dioxide'])
       total_sulfur_dioxide = float(request.form['total sulfur dioxide'])
       density = float(request.form['density'])
       pH = float(request.form['pH'])
       sulphates = float(request.form['sulphates'])
       alcohol = float(request.form['alcohol'])

       x = [[volatile_acidity,citric_acid,chlorides,sulphates,alcohol]]
       
       quality = model.predict(x)
       
       if quality == 3 or quality == 4:
           return render_template('index.html',prediction_text = "Quality of your wine is "+ str(quality) +" which is low")
       elif quality == 5 or quality == 6:
           return render_template('index.html',prediction_text = "Quality of your wine is "+ str(quality) +" which is fine")
       else:
           return render_template('index.html',prediction_text = "Quality of your wine is "+ str(quality) +" which is good")   
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)