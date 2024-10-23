import pickle

from flask import Flask
from flask import request # Process incoming request via POST
from flask import jsonify # Send back response in JSON format

dv_file = "dv.bin"
# model_file = 'model1.bin'
model_file = 'model2.bin'

with open(dv_file, 'rb') as d_in:
    dv = pickle.load(d_in)

with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)

app = Flask('hw5')

## in order to send the customer information we need to POST its data.
@app.route('/predict', methods=['POST'])
def predict():
    ## web services work best with json frame, 
    ## So after the user post its data in json format we need to access the body of json.
    client = request.get_json()

    # Transform customer info with DictVectorizer
    X = dv.transform([client])
    subscription = model.predict_proba(X)[0, 1]

    ## we need to cast numpy float type to python native float type
    ## same for subscription, casting the value using bool method
    result = {
        'subscription_probability': float(subscription)
    }
    ## send back the data in json format to the user
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)