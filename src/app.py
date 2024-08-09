from flask import Flask, request, jsonify
from model import *


app = Flask(__name__)


MODEL_AND_TOKENIZER = model_fn(model_dir="sentence-transformers/all-MiniLM-L12-v2")


@app.route('/user_query_tag_search', methods=['POST'])
def user_query_tag_search():
    data = request.json
    
    if 'inputs' not in data:
        return jsonify({"error": "Missing 'inputs' in request"})
    if 'index' not in data:
        return jsonify({"error": "Missing 'index' in request"})
    if 'topK' not in data:
        return jsonify({"error": "Missing 'topK' in request"})
    
    query = data.pop("inputs")
    query_index = data.pop("index")
    query_topK = int(data.pop("topK"))
    
    if query_index not in LOCAL_TAG_INDEX:
        return jsonify({"error": f"Expect index be in {LOCAL_TAG_INDEX}. But get {query_index}"})
    
    results = predict_topk_tags(query, query_index, query_topK, MODEL_AND_TOKENIZER)
    
    return jsonify(results)


@app.route('/user_query_tui_poi_search', methods=['POST'])
def user_query_tui_poi_search():
    data = request.json
    
    if 'inputs' not in data:
        return jsonify({"error": "Missing 'inputs' in request"})
    if 'topK' not in data:
        return jsonify({"error": "Missing 'topK' in request"})
    
    query_columns = None
    if 'columns' in data:
        query_columns =  data.pop("columns")
    
    query = data.pop("inputs")
    query_topK = int(data.pop("topK"))
    
    results = predict_topk_pois(query, query_topK, MODEL_AND_TOKENIZER, query_columns)
    
    return jsonify(results)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)