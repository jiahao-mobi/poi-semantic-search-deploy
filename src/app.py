from flask import Flask, request, jsonify
from model import *


app = Flask(__name__)

model_and_tokenizer = model_fn(model_dir="sentence-transformers/all-MiniLM-L12-v2")

@app.route('/user_query_tag_search', methods=['POST'])
def user_query_tag_search():
    data = request.json
    
    if 'inputs' not in data:
        return jsonify({"error": "Missing 'inputs' in request"})
    if 'index' not in data:
        return jsonify({"error": "Missing 'index' in request"})
    if 'topK' not in data:
        return jsonify({"error": "Missing 'topK' in request"})
    
    # Tokenize sentences
    query = data.pop("inputs")
    query_index = data.pop("index")
    query_topK = int(data.pop("topK"))
    
    if query_index not in LOCAL_INDEX:
        return jsonify({"error": f"Expect index be in {LOCAL_INDEX}. But get {query_index}"})
    
    results = predict_fn(query, query_index, query_topK, model_and_tokenizer)
    
    return jsonify(results)




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)