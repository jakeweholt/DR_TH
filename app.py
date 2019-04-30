from flask import Flask, request, jsonify
import json
import pandas as pd
from pandas.io.json import json_normalize
import pickle
from flask_configs import model_version, model_type
app = Flask(__name__)



def data_formatter(data):
    if isinstance(data, pd.core.series.Series):
        return pd.DataFrame(data).transpose()
    else:
        return data


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            data = json_normalize(request.json)
            transformed_data = data_processer.transform(data_formatter(data))
            return 'is_bad probability: {} \n'.format(model.predict_proba(transformed_data)[0][1])
        except KeyError:
            return "KeyError, data likely formatted incorrectly.\n"


@app.before_first_request
def load_model():
    global model
    global data_processer
    save_directory = 'saved_models/{}/{}/'.format(model_type, model_version)
    data_processer = pickle.load(open(save_directory + 'data_processer.p', 'rb'))
    model = pickle.load(open(save_directory + 'model_trained_on_validation_split.p', 'rb'))


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)
