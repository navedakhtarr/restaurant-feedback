import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

filename = 'multinb_model.pkl'
model = pickle.load(open(filename, 'rb'))
cv = pickle.load(open('cv_fit.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction', methods = ['POST'])
def predict():
    if request.method == 'POST':
    	message = request.form['message']
    	data = [message]
    	vect = cv.transform(data).toarray()
    	my_prediction = model.predict(vect)
    	return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)