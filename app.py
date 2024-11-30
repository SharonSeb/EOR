from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

from sklearn.preprocessing import StandardScaler

application = Flask(__name__)     # entry point, flask app

app = application

#Route for homepage

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data = CustomData(
            salt1 = request.form.get('salt1'),
            salt2 = request.form.get('salt2'),
            year = request.form.get('year'),
            f1hs = request.form.get('f1hs'),
            f2ls = request.form.get('f2ls'),

        )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', result=results[0])
    
 
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)