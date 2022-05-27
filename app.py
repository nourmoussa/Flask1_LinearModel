# import joblib
# import os
# from flask import Flask, jsonify, url_for, send_from_directory

# from flask import request
# from server import predict

# app = Flask(__name__)
# model = joblib.load('reg_1.pkl')


# @app.route('/', methods=['POST'])
# def predict():

#     # Get the data from the POST request.
#     data = request.get_json(force=True)
#     data = [[data["age"], data["gender"], data["height"], data["weight"], data["smoke"], data["alco"], data["active"]]]

#     # Make prediction using model loaded from disk as per the data.
#     prediction = model.predict(data)
#     # Take the first value of prediction
#     output = prediction[0]

#     return {"result": int(output)}

# # @app.route('/favicon.ico')
# # def favicon():
# #     return send_from_directory(os.path.join(app.root_path, 'static'),
# #                                'favicon.ico', mimetype='image/vnd.microsoft.icon')

# if __name__ == '__main__':
#     app.run()

from flask import Flask
import joblib
from flask import request

app = Flask(__name__)
model = joblib.load('reg_1.pkl')


@app.route('/', methods=['POST'])
def predict():
    # Get the data from the POST request.
    data = request.get_json(force=True)
    data = [[data["age"], data["gender"], data["height"], data["weight"], data["smoke"], data["alco"], data["active"]]]

    # Make prediction using model loaded from disk as per the data.
    # Take the first value of prediction
    prediction = model.predict(data)
    # Take the first value of prediction
    output = prediction[0]

    return {"result": int(output)}


if __name__ == '__main__':
    app.run()