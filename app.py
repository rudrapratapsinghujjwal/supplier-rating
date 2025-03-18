

from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    
    # Process file
    df = pd.read_csv(filepath)  # Assuming CSV file
    
    if not all(col in df.columns for col in ['on_time_delivery', 'quality_score', 'pricing_consistency', 'customer_feedback']):
        return jsonify({"error": "Invalid file format"})
    
    # ML Model
    X = df[['on_time_delivery', 'quality_score', 'pricing_consistency', 'customer_feedback']]
    y = np.random.rand(len(df)) * 5  # Dummy ratings for now
    model = LinearRegression()
    model.fit(X, y)
    df['rating'] = model.predict(X)
    
    return jsonify({"message": "File processed successfully", "ratings": df['rating'].tolist()})


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Use Render's assigned port or default to 10000
    app.run(host="0.0.0.0", port=port, debug=True)

