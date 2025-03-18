from flask import Flask, request, jsonify, render_template, send_file
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

def detect_columns(df):
    """Automatically detect relevant columns based on keyword matching."""
    col_mappings = {
        'on_time_delivery': ['on_time_delivery', 'delivery_time', 'timeliness'],
        'quality_score': ['quality_score', 'quality', 'product_quality'],
        'pricing_consistency': ['pricing_consistency', 'price_variation', 'cost_stability'],
        'customer_feedback': ['customer_feedback', 'reviews', 'satisfaction'],
    }
    
    detected_cols = {}
    for key, synonyms in col_mappings.items():
        for col in df.columns:
            if col.lower().replace(' ', '_') in synonyms:
                detected_cols[key] = col
                break
    
    return detected_cols

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
    df = pd.read_csv(filepath)
    detected_cols = detect_columns(df)
    
    if len(detected_cols) < 4:
        return jsonify({"error": "Invalid file format. Ensure columns related to delivery, quality, pricing, and feedback are present."})
    
    # Use detected columns
    X = df[list(detected_cols.values())]
    X = X.fillna(X.mean())  # Handle missing values
    
    # Dummy Rating Calculation (Improved)
    y = np.random.rand(len(df)) * 5  # Dummy ratings for now (0-5 scale)
    model = LinearRegression()
    model.fit(X, y)
    df['rating'] = model.predict(X)
    
    processed_filepath = os.path.join(app.config['PROCESSED_FOLDER'], 'processed_' + file.filename)
    df.to_csv(processed_filepath, index=False)
    
    return jsonify({
        "message": "File processed successfully",
        "ratings": df[['rating']].round(2).to_dict(orient='records'),
        "download_link": f"/download/{os.path.basename(processed_filepath)}"
    })

@app.route('/download/<filename>')
def download_file(filename):
    filepath = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    return send_file(filepath, as_attachment=True)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Use Render's assigned port or default to 10000
    app.run(host="0.0.0.0", port=port, debug=True)
