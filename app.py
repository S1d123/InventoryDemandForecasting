from flask import Flask, jsonify, render_template, request
import joblib
import os
import numpy as np

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/prediction")
def prediction():
    return render_template("prediction.html")

@app.route("/chart")
def chart():
    return render_template("chart.html")

# @app.route("/performance")
# def performance():
#     return render_template("performance.html")

@app.route('/prediction/predict',methods=['POST','GET'])
def result():

    item_weight= float(request.form['item_weight'])
    item_fat_content=float(request.form['item_fat_content'])
    item_visibility= float(request.form['item_visibility'])
    item_type= float(request.form['item_type'])
    item_mrp = float(request.form['item_mrp'])
    outlet_establishment_year= float(request.form['outlet_establishment_year'])
    outlet_size= float(request.form['outlet_size'])
    outlet_location_type= float(request.form['outlet_location_type'])
    outlet_type= float(request.form['outlet_type'])

    X= np.array([[ item_weight,item_fat_content,item_visibility,item_type,item_mrp,
                  outlet_establishment_year,outlet_size,outlet_location_type,outlet_type ]])

    scaler_path= r'D:\ML\BigMart-Sales-Prediction-With-Deployment\models\sc.sav'

    sc=joblib.load(scaler_path)

    X_std= sc.transform(X)

    model_path=r'D:\ML\BigMart-Sales-Prediction-With-Deployment\models\lr.sav'

    model= joblib.load(model_path)

    Y_pred=model.predict(X_std)
    prediction = int(Y_pred/X[0][4])
    # return jsonify({'Prediction': float(Y_pred/X[0][4])})
    return render_template(
        'output.html', 
        prediction=prediction, 
        title='Employee Attrition Prediction'
        )

if __name__ == "__main__":
    app.run(debug=True, port=9457)
