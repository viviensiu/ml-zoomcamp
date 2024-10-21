import pickle

from flask import Flask
from flask import request # Process incoming request via POST
from flask import jsonify # Send back response in JSON format


model_file = 'model_C=1.0.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('churn')

## in order to send the customer information we need to POST its data.
@app.route('/predict', methods=['POST'])
def predict():
    ## web services work best with json frame, 
    ## So after the user post its data in json format we need to access the body of json.
    customer = request.get_json()

    # Transform customer info with DictVectorizer
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    churn = y_pred >= 0.5

    ## we need to cast numpy float type to python native float type
    ## same for churn, casting the value using bool method
    result = {
        'churn_probability': float(y_pred), 
        'churn': bool(churn)
    }
    ## send back the data in json format to the user
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)