"""
This script implements a simple Web API based on Flask,
in order to serve predictions for a classification model.
The model was trained with `classifier_train.py`.

This API uses the classifier and vectoriser saved during
training as pickle files.

It can be run from the terminal with:

    python classifier_predict.py
"""
import pickle
from flask import Flask
from flask import request
from flask import jsonify

app = Flask(__name__)

@app.route('/')
def hello_world():
    return "Hello, use the /predict endpoint"

@app.route('/predict', methods=['POST'])
def serve_prediction():
    # Get the request, e.g. {"document": "some text to classify here"}
    doc = request.json['document']
    # Load the vectoriser
    with open('my_vectoriser.pickle', 'rb') as f:
        vectoriser = pickle.load(f)
    # Load the classifier
    with open('my_classifier.pickle', 'rb') as f:
        classifier = pickle.load(f)

    # Vectorise the document
    # Note: only transform(), no need to fit
    # because we're using the vocabulary from the training set
    # Note 2: transform() expects a list of items
    vectorised_doc = vectoriser.transform([doc])

    # Make prediction
    pred = classifier.predict(vectorised_doc)

    # Output (note: pred is a list of one item)
    output = {
        "prediction": pred[0],
        "input_document": doc,
        "timestamp": "...now..."
    }
    # Serve prediction back to client
    return jsonify(output)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5555, debug=True)
