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
    # Error handling for missing keys
    expected_keys = ["qubit_frequency", "simulation_mode", "driving_frequency", "duration", "rabi_rate", "t1", "t2", "shots"] 
    data = request.get_json()
    missing_keys = [key for key in expected_keys if key not in data]
    if missing_keys:
        return jsonify({"error": f"Missing keys: {missing_keys}"}), 400
    
    # Parsing data from POST request into relevant variables
    wq = request.get_json()["qubit_frequency"]
    simulation_mode = request.get_json()["simulation_mode"] 
    wd = request.get_json()["driving_frequency"]
    t = request.get_json()["duration"]   
    rabi_rate = request.get_json()["rabi_rate"]
    t1 = request.get_json()["t1"]
    t2 = request.get_json()["t2"]
    shots = request.get_json()["shots"]

    # Returns JSON data (this will be redundant to returning the initial dictionary but I am just checking it is working correctly)
    return jsonify({"qubit_frequency": wq, "simulation_mode": simulation_mode, "driving_frequency": wd, "duration": t, "rabi_rate": rabi_rate, "t1": t1, "t2": t2, "shots": shots})

# Driver 
if __name__ == '__main__':
    app.run(debug=True)

