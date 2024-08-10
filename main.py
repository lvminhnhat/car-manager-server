import pandas as pd
import csv
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

dataLink= "https://docs.google.com/spreadsheets/d/e/2PACX-1vRBF-LTEigaGsW4jI03UfEDuBjbKSJNYzLgcq71Q8oTec0EaW4ZhtLQ7ESzSLY-bvL-w8rDLVZJv9ta/pub?output=csv"

df = pd.read_csv(dataLink)
df = df.iloc[:, 1:]
df.columns = ['Nâng Kính xe', 'Hạ kính xe', 'Dừng nâng/hạ kính', 'Đặt kính ở chế độ nhìn xuyên thấu', 'Ngăn chặn người ngoài nhìn xuyên qua kính']
# In dữ liệu ra
print(df.head())


# Chuyển đổi dữ liệu thành định dạng dài
df_long = df.melt(var_name='title', value_name='command')
df_long = df_long.dropna()  # Xóa các giá trị thiếu

# Tạo bộ dữ liệu huấn luyện và kiểm tra
X_train, X_test, y_train, y_test = train_test_split(df_long['command'], df_long['title'], test_size=0.2, random_state=42)

# Tạo pipeline mô hình
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', MultinomialNB())
])

# Huấn luyện mô hình
model.fit(X_train, y_train)

# Dự đoán và đánh giá mô hình
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Ví dụ dự đoán cho một câu lệnh mới
new_command = ["nhìn đéo gì"]
predicted_title = model.predict(new_command)
predicted_proba = model.predict_proba(new_command)

print(f"Câu lệnh '{new_command[0]}' thuộc về tiêu đề: {predicted_title[0]}")
print("Xác suất dự đoán:")
for i, title in enumerate(model.classes_):
    print(f"{title}: {predicted_proba[0][i]:.4f}")

import getpass
import os
import threading

from flask import Flask, request, jsonify
app = Flask(__name__)
portOpen = 5060


@app.route('/', methods=['POST']) # Changed route name
def index(): # Changed function name to avoid conflict
    # Nhận dữ liệu JSON từ yêu cầu
    data = request.get_json()
    new_command = data.get('command')

    if not new_command:
        return jsonify({'error': 'No command provided'}), 400

    # Dự đoán tiêu đề và xác suất
    predicted_proba = model.predict_proba([new_command])[0]
    
    # Tìm xác suất cao nhất
    max_prob = max(predicted_proba)
    predicted_title = model.classes_[predicted_proba.argmax()]

    # Nếu xác suất cao nhất thấp hơn 0.5, trả về "Uncertain"
    if max_prob < 0.5:
        predicted_title = 'Uncertain'

    result = {
        'predicted_title': predicted_title,
        'probabilities': {title: prob for title, prob in zip(model.classes_, predicted_proba)}
    }

    return jsonify(result)

app.run()
