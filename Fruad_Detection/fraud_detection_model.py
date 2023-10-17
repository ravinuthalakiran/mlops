# fraud_detection_model.py
import logging
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import io

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


app = Flask(__name__)

logging.info('Loading and preprocessing dataset.')


# Load and preprocess the dataset
df = pd.read_csv('creditcard.csv')
class_0_df = df[df.Class==0].head(1000)
class_1_df = df[df.Class==1]

result_df = pd.concat([class_0_df, class_1_df], axis=0).reset_index(drop=True)
df = result_df.copy()
# ... your preprocessing steps here ...
df - df.iloc[:,28:]

logging.info('Dataset loaded and preprocessed.')


# Train the Random Forest Classifier model
model = RandomForestClassifier()
X = df.drop(['Class'], axis=1)  # Assuming 'Class' is the target variable
y = df['Class']
model.fit(X, y)

# Save the trained model
joblib.dump(model, 'fraud_detection_model.pkl')

logging.info('Model trained and saved.')

# Load the saved model
model = joblib.load('fraud_detection_model.pkl')

@app.route('/score', methods=['POST'])
def score():
    try:
        logging.info('Scoring request received.')
        if request.content_type == 'application/json':
            data = request.json
            df = pd.DataFrame([data])
        elif request.content_type == 'text/csv':
            csv_file = request.files['file']
            df = pd.read_csv(csv_file)
        elif request.content_type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
            excel_file = request.files['file']
            df = pd.read_excel(io.BytesIO(excel_file.read()), sheet_name='Sheet1')
        else:
            logging.error(f'An error occurred: {str(e)}')
            return jsonify({'error': 'Unsupported content type'})

        # Preprocessing (if any)
        # ... your preprocessing steps here ...

        # Make predictions
        pred = model.predict(df)

        # Return the prediction
        logging.info('Scoring completed.')
        return jsonify({'prediction': pred.tolist()})
    except Exception as e:
        logging.error(f'An error occurred: {str(e)}')
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

