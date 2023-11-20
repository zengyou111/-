from flask import Flask, request, jsonify
from flask_cors import CORS  # 导入 CORS 扩展
from gevent import pywsgi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)
CORS(app)  # 在 Flask 应用中启用 CORS

def calculate_similarity(content1, content2):
    documents = [content1, content2]

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return similarity_matrix[0][1]

@app.route('/code-similarity-content', methods=['POST'])
def code_similarity_content():
    data = request.get_json()

    content1 = data.get('content1', '')
    content2 = data.get('content2', '')

    if content1 and content2:
        similarity = calculate_similarity(content1, content2)
        return jsonify({'result': similarity})

    return jsonify({'error': 'Content not provided'}), 400

if __name__ == '__main__':
    server = pywsgi.WSGIServer(('192.168.79.1', 5000), app)
    server.serve_forever()
