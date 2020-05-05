# Importing necessary libraries

import numpy as np
from flask import Flask, request, make_response
import json
import pickle
from flask_cors import cross_origin


# Declaring the flask app

app = Flask(__name__)

# Loading the modle from pickle file
model = pickle.load(open('rf.pkl', 'rb'))


# getting and sending response to dialogflow
@app.route('/webhook', methods=['POST'])
@cross_origin()
def webhook():
    req = request.get_json(silent=True, force=True)
    res = processRequest(req)
    res = json.dumps(res, indent=4)
    r = make_response(res)
    r.headers['Content-Type'] = 'application/json'
    return r


# Processing the request from dialogflow

def processRequest(req):
    
    result = req.get("queryResult")

    # Fetching the data points
    parameters = result.get("parameters")
    Petal_length = parameters.get("number")
    Petal_width = parameters.get("number1")
    Sepal_length = parameters.get("number2")
    Sepal_width = parameters.get("number3")

    int_features = [Petal_length, Petal_width, Sepal_length, Sepal_width]

    # Dumping the data into a array
    final_features = [np.array(int_features)]

    # Getting the intent which has fullfilment enabled
    intent = result.get("intent").get('displayName')

    # Fitting out the model with the data points
    if (intent=="IrisBot"):
        prediction = model.predict(final_features)
        
        output = round(prediction[0], 2)

        if output == 0:
            flowr = 'Setosa'

        if output == 1:
            flowr = 'Versicolour'
        
        if output == 2:
            flowr = 'Virginica'

        # Returning back to the fullfilment text back to the dialogflow
        fullfillmentText = "The Iris type seems to be.. {}!".format(flowr)

        #log.write_log(sessionID, "Bot Says: "+fulfillmentText)

        return {
            "fullfillmentText": fullfillmentText
        }

if __name__ == "__main__":
    app.run()