from flask import Flask
from flask import request
from waitress import serve

print('Initializing model...')
from model_initialization import model, predict
print('Initialization completed.')

print("Starting server...")
app = Flask(__name__)
print("Server started.")


@app.route("/api/label", methods=['POST'])
def predict_text_label():

    try:

        data = request.json['data']
        predict_val = predict(model, data)

        return {'initial_data': data, 'predict': predict_val}

    except Exception:

        return {'error': 'Incorrect data'}, 400


if __name__ == "__main__":
    serve(app, port=8000)
