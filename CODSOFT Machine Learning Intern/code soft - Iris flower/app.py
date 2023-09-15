import pandas as pd
import numpy as np 
df = pd.read_csv(r'https://raw.githubusercontent.com/amankharwal/Website-data/master/IRIS.csv')

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

x = df.drop(columns='species')
y = df['species']
x_train, x_test, y_train , y_test = train_test_split(x,y, test_size=0.2, random_state=10)
lor = LogisticRegression()

lor.fit(x_train, y_train)

y_pred = lor.predict(x_test)


from flask import Flask, render_template, url_for, request, redirect

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods = ['GET','POST'])
def predict ():
    if request.method=='GET':
        return render_template("home.html")
    else:
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_length'])

        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
        result = lor.predict(input_data)
        
        return render_template('home.html', result = result)
    

@app.route('/home')
def predi():
    return render_template('home.html')
if __name__== "__main__":
    app.run(debug=True) 
    