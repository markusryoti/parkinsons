from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from PIL import Image
import torch
from parkinsons import ParkinsonModel, do_predict

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.DEBUG)

model = ParkinsonModel()
model.load_state_dict(torch.load('model.pt'))
model.eval()


@app.route('/predict', methods=['POST'])
def predict():

    if request.method == 'POST':
        img_data = request.files.get('img')
        app.logger.info('form data: %s', img_data)

        img = Image.open(img_data)
        img = img.convert('RGB')

        pred = do_predict(model, img)

        return jsonify(pred)


app.run(host='localhost', port=5000)
