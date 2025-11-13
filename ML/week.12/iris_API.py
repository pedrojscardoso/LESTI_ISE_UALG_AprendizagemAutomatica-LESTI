#import Orange
import pickle
import numpy as np
from flask import Flask, jsonify, Response
import logging

# define logging level
logging.basicConfig(level=logging.INFO)

# load model created with Orange (you might need to change the path)
model = pickle.load(open("./ML/week.12/model_tree.pkcls", "rb"))

# get the classes names
classes = model.original_domain.class_var.values # ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
logging.info(f"Classes: {classes}")

# get the features names
columns = [var.name for var in model.domain.attributes]  # ['sepal length', 'sepal width', 'petal length', 'petal width']
logging.info(f"Columns: {columns}")

# ------------------------------------
# API: we will use Flask to create a simple API
app = Flask(__name__)

# define the route for the API
# http://<host>:<port>/usage
# returns the usage of the API
@app.route("/usage")
@app.route("/")
def usage():
    return jsonify(
        {
            "route_order": "/".join(columns),
            "usage example": "http://<host>:<port>/" + "/".join(map(str, [5.1,	3.5, 1.4,0.2]))
        }
    )

# define the route which will be used to predict
@app.route("/<a>/<b>/<c>/<d>", methods=["GET"])
def predict(a, b, c, d):
    # gets the values from the URL and converts them to float and creates a numpy array (2D array, i.e., [[...]])
    instance = np.array([[a, b, c, d]], dtype=float)
    
    # use the already trained model to predict the class for the given instance
    # we need to convert the result to a list (it is a numpy array)
    pred = model.predict(instance)
    logging.info(f"Prediction: {pred}")
    pred = pred.tolist()[0]

    # gets the index of the class with the highest probability
    idx = np.argmax(pred)

    # returns the result
    return jsonify(
        {
            "class": classes[idx],
            "prob": pred[idx],
            "details": {
                "classes": classes,
                "probs": pred
            },
            "code": 200  # para ajudar na verificação de um "200 - OK"
        }
    ), 200


# handles missing routes (404)
@app.errorhandler(404)
def not_found(e):
    return jsonify(
        {
            "code": e.code,
            "usage": usage().json
        }
    ), 404


if __name__ == "__main__":
    # run the API
    app.run(
        host='0.0.0.0',  # needed to access from outside the container
        port=5002,  # define the port
        debug=True # e.g., restarts the API when the code changes
    )

#%%
