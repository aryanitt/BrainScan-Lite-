from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        data = CustomData(
            Q1=int(request.form.get('Q1', 0)),
            Q2=int(request.form.get('Q2', 0)),
            Q3=int(request.form.get('Q3', 0)),
            Q4=int(request.form.get('Q4', 0)),
            Q5=int(request.form.get('Q5', 0)),
            Q6=int(request.form.get('Q6', 0)),
            Q7=int(request.form.get('Q7', 0)),
            Q8=int(request.form.get('Q8', 0)),
            Q9=int(request.form.get('Q9', 0)),
            Q10=int(request.form.get('Q10', 0))
        )

        pred_df = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        return render_template('index.html', results=results[0])

if __name__ == "__main__":
    app.run(debug=True)
