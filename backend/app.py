# Qublitz Backend API
# Author: Neo Cai
# Date: 2021-06-06
# Version: 0.1

# Importing the libraries
from flask import Flask, jsonify, request
from flask_restful import Resource, Api

# Initializing Flask app
app = Flask(__name__)

# Creating API Object with POST method
@app.route('/', methods = ['POST'])
def post():
    data = request.get_json() # Gets JSON data from POST request
    return jsonify(data) # Returns that data in Json format 

# Driver 
if __name__ == '__main__':
    app.run(debug=True)

