from flask import Flask
from flask import request
from flask import Response
from joblib import load
import jsonpickle
import numpy as np


app = Flask(__name__)
@app.route("/prediction", methods=['POST'])
def make_prediction():
    """
    This function is called by a POST request to the "/prediction"
    API endpoint. It uses a pre-trained model to make a prediction
    of liver or no liver disease for an input feature vector.

    Parameters:
    - POST request: See rest_client.py for command line arguments

    Returns:
    - Response(Response): A requests Response object including
    pickled json containing response and status code.

    """
    data = jsonpickle.decode(request.data)
    try:
        X_test = np.array(data["features"],
                          dtype=np.float64).reshape(1, 11)
        loaded_model = load("best_tree.pkl")
        pred = loaded_model.predict(X_test)
        response = {"patient_id": data["id"],
                    "prediction": pred[0]}
        return Response(response=jsonpickle.encode(response),
                        mimetype="application/json",
                        status=200)

    except Exception as inst:
        response = {"Error": inst}
        return Response(response=jsonpickle.encode(response),
                        mimetype="application/json",
                        status=500)


app.run(host="0.0.0.0", port=5000)
