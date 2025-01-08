import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods = ['POST'])
def predict():
    message = request.form['message']
    model = pickle.load(open('model.pkl','rb'))
    tfidf = pickle.load(open('tfidf.pkl','rb'))
    text = tfidf.transform([message])
    prediction = model.predict(text)
    message = " "
    if prediction:
        message = 'The text is likely wirtten by AI'
    else:
        message = "The text is likely written by Human"
    return render_template('home.html', prediction_text=message)


if __name__ == '__main__':
    app.run(debug=True)