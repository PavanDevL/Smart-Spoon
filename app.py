from flask import Flask, render_template, request, redirect, session, jsonify, send_file
import os
import pandas as pd
import random
import string
import json
import plotly.express as px
import pandas as pd
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
import numpy as np
from collections import Counter
import seaborn as sns
import plotly.graph_objs as go


app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Folder to save uploads
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

survey_file_path = './smart_spoon_survey_synthetic_1000.csv'

# In-memory feedback list (or you can save to database)
feedback_list = []

# Simple "in-memory" user database
users = {}

# Dummy password reset tokens
reset_tokens = {}

# Load your trained CNN model
MODEL_PATH = 'best_sodium_model_finetuned.h5'
model = load_model(MODEL_PATH)

# Define your class labels (example)
class_labels = ['Low Sodium Food', 'High Sodium Food', 'Balanced Diet', 'High Sugar Food']


# Helpers
def load_users():
    if not os.path.exists('users.json'):
        return {}
    with open('users.json', 'r') as f:
        return json.load(f)

def save_users(users):
    with open('users.json', 'w') as f:
        json.dump(users, f)

def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        return 'Positive'
    elif polarity < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

def generate_wordcloud(feedbacks):
    text = " ".join(feedbacks)
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    img = BytesIO()
    wc.to_image().save(img, format='PNG')
    img.seek(0)
    return img

# =================== Authentication Routes ===================

@app.route('/')
def index():
    return redirect('/login')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        users = load_users()
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username]['password'] == password:
            session['username'] = username
            return jsonify({'status': 'success', 'message': 'Login successful!'})
        else:
            return jsonify({'status': 'error', 'message': 'Invalid username or password'})
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        users = load_users()
        username = request.form['username']
        password = request.form['password']
        security_answer = request.form['security_answer']
        if username in users:
            return jsonify({'status': 'error', 'message': 'User already exists!'})
        users[username] = {'password': password, 'security_answer': security_answer}
        save_users(users)
        return jsonify({'status': 'success', 'message': 'Signup successful!'})
    return render_template('signup.html')

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        users = load_users()
        username = request.form['username']
        security_answer = request.form['security_answer']
        new_password = request.form['new_password']
        if username in users and users[username]['security_answer'].lower() == security_answer.lower():
            users[username]['password'] = new_password
            save_users(users)
            return jsonify({'status': 'success', 'message': 'Password reset successful!'})
        else:
            return jsonify({'status': 'error', 'message': 'Incorrect details!'})
    return render_template('forgot_password.html')

@app.route('/reset_password', methods=['POST'])
def reset_password():
    token = request.form['token']
    new_password = request.form['new_password']
    if token in reset_tokens:
        username = reset_tokens.pop(token)
        users[username] = new_password
        return redirect('/login')
    else:
        return render_template('forgot_password.html', error="Invalid or expired token")

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect('/login')

# =================== Main Pages ===================

@app.route('/home')
def home():
    if 'username' not in session:
        return redirect('/login')
    return render_template('home.html')

@app.route('/survey')
def survey_insights():
    if 'username' not in session:
        return redirect('/login')
    return render_template('survey.html')

@app.route('/sentiment_analysis', methods=['GET', 'POST'])
def sentiment_analysis():
    if 'username' not in session:
        return redirect('/login')

    if request.method == 'POST':
        feedback = request.form['feedback']
        if feedback.strip():
            feedback_list.append(feedback)

    return render_template('sentiment.html')

# =================== File Upload and Prediction ===================

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    filename = file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    return "File uploaded successfully", 200

@app.route('/predict', methods=['POST'])
def predict():
    predictions = []
    for filename in os.listdir(UPLOAD_FOLDER):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filepath = os.path.join(UPLOAD_FOLDER, filename)

            # Load and preprocess image
            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)  # batch size = 1
            img_array = img_array / 255.0  # normalize

            # Predict
            preds = model.predict(img_array)
            predicted_class = class_labels[np.argmax(preds)]

            predictions.append({'filename': filename, 'predicted_class': predicted_class})

    # Save predictions to CSV
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(os.path.join(UPLOAD_FOLDER, 'predictions.csv'), index=False)

    return jsonify({'status': 'success', 'predictions': predictions})

@app.route('/download_results')
def download_results():
    path = os.path.join(UPLOAD_FOLDER, 'predictions.csv')
    return send_file(path, as_attachment=True)

@app.route('/get_survey_data')
def get_survey_data():
    df = pd.read_csv(survey_file_path)
    pie_data = df['How often do you consume outside food?'].value_counts()
    pie_lables = pie_data.index
    pie = []
    pie_l = []
    for label in pie_lables:
        pie.append(int(pie_data[label]))
        pie_l.append(label)
    bar_data = df['Salt1'].value_counts()
    bar_labels = bar_data.index
    bar_d = []
    bar_l = []
    for label in bar_labels:
        bar_d.append(int(bar_data[label]))
        bar_l.append(label)
    survey_data = {
        "pie": {
            #"labels": ['Satisfied', 'Neutral', 'Dissatisfied'],
            "labels": pie_l,
            "data": pie
        },
        "bar": {
            "labels": bar_l,
            "data": bar_d
        },
        "line": {
            "labels": ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
            "data": [3.5, 3.8, 4.0, 4.2, 4.4, 4.5]
        }
    }

    return jsonify(survey_data)

@app.route('/get_sentiment_data')
def get_sentiment_data():
    df = pd.read_csv(survey_file_path)

    # Combine existing feedback + new feedback
    existing_feedbacks = df['Any suggestions/comments?'].dropna().tolist()
    all_feedbacks = existing_feedbacks + feedback_list

    sentiments = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
    for fb in all_feedbacks:
        sentiment = analyze_sentiment(fb)
        sentiments[sentiment] += 1

    return jsonify(sentiments)

@app.route('/wordcloud')
def wordcloud_img():
    df = pd.read_csv(survey_file_path)

    existing_feedbacks = df['Any suggestions/comments?'].dropna().tolist()
    all_feedbacks = existing_feedbacks + feedback_list

    img = generate_wordcloud(all_feedbacks)
    return send_file(img, mimetype='image/png')


# =================== Error Handler ===================

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

# =================== Main ===================

if __name__ == '__main__':
    app.run(debug=True)
