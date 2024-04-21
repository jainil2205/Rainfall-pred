import os

# -------------------------------------------------model_code------------------------------------------------------------
import sqlite3

conn = sqlite3.connect('rainfall_database')
cur = conn.cursor()
try:
    cur.execute('''CREATE TABLE user (
     name varchar(20) DEFAULT NULL,
      email varchar(50) DEFAULT NULL,
     password varchar(20) DEFAULT NULL,
     gender varchar(10) DEFAULT NULL,
     age int(11) DEFAULT NULL
   )''')

except:
    pass



# include packages
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
import sklearn

# reading the dataset
dataset = pd.read_csv("Daily Rainfall dataset.csv")
dataset.head()

from sklearn.model_selection import train_test_split

predictors = dataset.drop(["year", "Rainfall"], axis=1)
target = dataset["Rainfall"]

X_train, X_test, Y_train, Y_test = train_test_split(predictors, target, test_size=0.20, random_state=0)

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

lr = LinearRegression()

lab_enc = preprocessing.LabelEncoder()
Y_train = lab_enc.fit_transform(Y_train)

lr.fit(X_train, Y_train)

Y_pred_lr = lr.predict(X_test)

score_lr = lr.score(X_test, Y_test)
print("The accuracy score achieved using Logistic regression is: " + str(score_lr) + " %")

data_1 = pd.read_csv('rainfall in india 1901-2015.csv')
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

SUBDIVISION = le.fit_transform(data_1.SUBDIVISION)
data_1['SUBDIVISION'] = SUBDIVISION

data_1.dropna(inplace=True)

data_1['Flood'].replace(['YES', 'NO'], [1, 0], inplace=True)
x1 = data_1.iloc[:, 0:14]

y1 = data_1.iloc[:, -1]

from sklearn import model_selection, neighbors
from sklearn.model_selection import train_test_split

x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.2)

from sklearn.naive_bayes import GaussianNB

clf_NB = GaussianNB()
clf_NB.fit(x1_train, y1_train)
y_pred_NB = clf_NB.predict(x1_test)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_pred_NB, y1_test))
score_nb = accuracy_score(y_pred_NB, y1_test)

from flask import Flask, render_template, url_for, request, flash, redirect, session

app = Flask(__name__)
app.config['SECRET_KEY'] = '881e69e15e7a528830975467b9d87a98'


# -------------------------------------home_page-------------------------------------------------------------------------

@app.route('/')
@app.route('/home')
def home():
    if not session.get('logged_in'):
        return render_template('home.html')
    else:
        return redirect(url_for('user_account'))


# -------------------------------------about_page-------------------------------------------------------------------------
@app.route("/about")
def about():
    return render_template('about1.html')


# -------------------------------------about_page-------------------------------------------------------------------------



# --------------------------------------help_page-------------------------------------------------------------------------
@app.route("/helping")
def helping():
    return render_template('help.html')


# --------------------------------------help_page-------------------------------------------------------------------------

# -------------------------------------user_login_page-------------------------------------------------------------------------
@app.route('/user_login', methods=['POST', 'GET'])
def user_login():
    conn = sqlite3.connect('rainfall_database')
    cur = conn.cursor()
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['psw']
        print('asd')
        count = cur.execute('SELECT * FROM user WHERE email = "%s" AND password = "%s"' % (email, password))
        print(count)
        # conn.commit()
        # cur.close()
        l = len(cur.fetchall())
        if l > 0:
            flash(f'Successfully Logged in')
            return render_template('user_account.html')
        else:
            print('hello')
            flash(f'Invalid Email and Password!')
    return render_template('user_login.html')


# -------------------------------------user_login_page-----------------------------------------------------------------

# -------------------------------------user_register_page-------------------------------------------------------------------------

@app.route('/user_register', methods=['POST', 'GET'])
def user_register():
    conn = sqlite3.connect('rainfall_database')
    cur = conn.cursor()
    if request.method == 'POST':
        name = request.form['uname']
        email = request.form['email']
        password = request.form['psw']
        gender = request.form['gender']
        age = request.form['age']
        cur.execute("insert into user(name,email,password,gender,age) values ('%s','%s','%s','%s','%s')" % (
        name, email, password, gender, age))
        conn.commit()
        # cur.close()
        print('data inserted')
        return redirect(url_for('user_login'))

    return render_template('user_register.html')


# -------------------------------------user_register_page-------------------------------------------------------------------------
import random

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    # Getting input values from the form
    day = float(request.form['day'])
    visibilityHigh = float(request.form['visibilityHigh'])
    visibilityAvg = float(request.form['visibilityAvg'])
    month = float(request.form['month'])
    tempHigh = float(request.form['tempHigh'])
    tempAvg = float(request.form['tempAvg'])
    visibilityLow = float(request.form['visibilityLow'])
    tempLow = float(request.form['tempLow'])
    windAvg = float(request.form['windAvg'])
    DPLow = float(request.form['DPLow'])
    DPHigh = float(request.form['DPHigh'])
    DPAvg = float(request.form['DPAvg'])
    humidityHigh = float(request.form['humidityHigh'])
    SLPHigh = float(request.form['SLPHigh'])
    SLPLow = float(request.form['SLPLow'])
    SLPAvg = float(request.form['SLPAvg'])
    humidityAvg = float(request.form['humidityAvg'])
    humidityLow = float(request.form['humidityLow'])
    
    global lr
    if request.method == 'POST':
        # Making the prediction
        predicted_rainfall = lr.predict([[month, day, tempHigh, tempAvg, tempLow, DPHigh,
                                           DPAvg, DPLow, humidityHigh, humidityAvg, humidityLow,
                                           SLPHigh, SLPAvg, SLPLow, visibilityHigh, visibilityAvg,
                                           visibilityLow, windAvg]])
        
        # Set specific rainfall values for each month
        month_rainfall = {
            1: 12.5,
            2: 22.7,
            3: 29.7,
            4: 39.2,
            5: 42.3,
            6: 87.92,
            7: 278.32,
            8: 254.9,
            9: 167.56,
            10: 49.43,
            11: 20.34,
            12: 14.3
        }
        
        # Check if the predicted rainfall is less than the specific value for the month
        if predicted_rainfall[0] < month_rainfall[int(month)]:
            # Adjust the predicted rainfall to the specific value for the month
            predicted_rainfall[0] = month_rainfall[int(month)]
        
        flash(f'The Rainfall is {predicted_rainfall[0]:.2f}mm')
        
        return render_template('user_account.html')

# ------------------------------------predict_page-----------------------------------------------------------------

@app.route("/flood")
def flood():
    return render_template('flood.html')


# -------------------------------------

@app.route('/predicts', methods=['POST', 'GET'])
def predicts():
    SUBDIVISION = request.form['SUBDIVISION']
    YEAR = request.form['YEAR']
    JAN = request.form['JAN']
    FEB = request.form['FEB']
    MAR = request.form['MAR']
    APR = request.form['APR']
    MAY = request.form['MAY']
    JUN = request.form['JUN']
    JUL = request.form['JUL']
    AUG = request.form['AUG']
    SEP = request.form['SEP']
    OCT = request.form['OCT']
    NOV = request.form['NOV']
    DEC = request.form['DEC']
    out = clf_NB.predict([[float(SUBDIVISION), float(YEAR), float(JAN), float(FEB), float(MAR), float(APR),
                           float(MAY), float(JUN), float(JUL), float(AUG), float(SEP), float(OCT),
                           float(NOV), float(DEC)]])
    print(out)
    if out[0] == 1:
        s = print('Yes floods in {}')
        print(s.format(SUBDIVISION))
        flash(f'Yes')
        return render_template('index.html')

    else:
        print('No')
        flash(f'No')
        return render_template('noFlood.html')

    return render_template('flood.html')


@app.route("/logout")
def logout():
    session['logged_in'] = False
    return home()


@app.route("/logoutd", methods=['POST', 'GET'])
def logoutd():
    return home()


if __name__ == '__main__':
    app.secret_key = os.urandom(12)
    app.run(debug=True)
