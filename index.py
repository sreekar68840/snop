import os
import re

from flask import Flask, request, jsonify
 
app = Flask(__name__)
 
@app.route('/getdata', methods=['POST'])
def query_api():
    data = request.get_json()
    user_question = data.get("user_question")
    return jsonify({
         "message": user_question
    })

if __name__ == '__main__':
    print("Starting Flask server on port 5000...")
    app.run(host="0.0.0.0", port=5000)
