from flask import Flask
from flask import render_template, request, Response, json
from models import model
from models import ultis
from time import time
from flask import Flask, request, render_template, jsonify
import time 
from io import BytesIO
import base64
import numpy as np
from PIL import Image
import cv2 
from flask_cors import CORS

SERVER_NAME = 'http://localhost:5000'
app = Flask(__name__)

Handwritten_model = model.Handwritten_Recognition()



@app.route("/handwritten", methods=['POST'])
def runCommentSemantic():
    start_time = time.time()
    data = {"success": False}

    if request.method == "POST":
        image = request.files.get("image", None)
        if image is not None:
            try:
                # Read image by Opencv
                image = ultis.read_image(image)

                # classify the input image
                temp = Handwritten_model.predict_image(image)

                data["output"] = int(np.argmax(temp))
                data["confidence"] = float(np.max(temp))
                # indicate that the request was a success
                data["success"] = True
            except Exception as ex:
                data['error'] = ex
                print(str(ex))
        else:
            image = request.form.get("image", None)

            if image is not None:
                try: 
                    # Read Image 
                    image = image.split("base64,")[1]
                    image = BytesIO(base64.b64decode(image))
                    image = Image.open(image) 
                    image = Image.composite(image, Image.new('RGB', image.size, 'white'), image)
                    image = image.convert('L')
                    image = image.resize((28, 28), Image.ANTIALIAS) 
                    image = 1 - np.array(image, dtype=np.float32) / 255.0

                    # classify the input image
                    temp = Handwritten_model.predict_image(image)

                    data["output"] = int(np.argmax(temp))
                    data["confidence"] = float(np.max(temp))
                    # indicate that the request was a success
                    data["success"] = True
                except Exception as ex:
                    data['error'] = ex
                    print(str(ex))

    data['run_time'] = "%.2f" % (time.time() - start_time)
    # return the data dictionary as a JSON response
    return jsonify(data)

@app.route('/', methods=['GET', 'POST'])
def main():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=3000)