from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Initialize Flask application
application = Flask(__name__)
app = application

## Route for the home page
@app.route('/')
def index():
    # Render the homepage template (index.html)
    return render_template('index.html')

# Route for handling predictions
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    # If the request method is GET, return the form page (home.html)
    if request.method == 'GET':
        return render_template('home.html')
    else:
        # Extract form data and map to the CustomData class for preprocessing
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),  # Swapped values here
            writing_score=float(request.form.get('reading_score'))   # Swapped values here
        )
        
        # Convert the data to a DataFrame for model prediction
        pred_df = data.get_data_as_data_frame()
        print(pred_df)  # Log the data to be predicted
        print("Before Prediction")

        # Initialize the prediction pipeline and make predictions
        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        print("After Prediction")
        
        # Return the result to be displayed on the home page
        return render_template('home.html', results=results[0])

# Run the app on 0.0.0.0 so it's available externally as well
if __name__ == "__main__":
    app.run(host="0.0.0.0")

