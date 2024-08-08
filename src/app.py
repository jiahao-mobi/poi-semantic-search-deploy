from flask import Flask, request, jsonify
from model import *


app = Flask(__name__)

model_and_tokenizer = model_fn(model_dir="sentence-transformers/all-MiniLM-L12-v2")

@app.route('/user_query_tag_search', methods=['POST'])
def hello():
    data = request.json
    
    if 'inputs' not in data:
        return jsonify({"error": "Missing 'inputs' in request"}), 400
    if 'index' not in data:
        return jsonify({"error": "Missing 'index' in request"}), 400
    if 'topK' not in data:
        return jsonify({"error": "Missing 'topK' in request"}), 400
    
    topk_tags = predict_fn(data, model_and_tokenizer)
    
    return jsonify(topk_tags)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)