from flask import Flask, Response, json, request, render_template
import os
import io
import base64
import boto3
import json
import random
import requests
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

class PredictForm(FlaskForm):
    image_url = StringField('Image', validators=[DataRequired()])
    threshold = StringField('Threshold', validators=[DataRequired()])
    submit = SubmitField('Predict')

app = Flask(__name__)
SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY
runtime = boto3.client('runtime.sagemaker')

# here is how we are handling routing with flask:
@app.route('/')
def index():
    form = PredictForm()
    return render_template('index.html', title='Ahoy!', form=form)

@app.route('/prediction', methods=["GET","POST"])
def user():
    resp_dict = {}
    if request.method == "GET":
        image_url = request.args.get('image_url')
        threshold = request.args.get('threshold')
    if request.method == "POST":
        data = request.form
        image_url = data.get('image_url','')
        threshold = data.get('threshold','')
    threshold = float(threshold)

    input_file = 'test.jpg'
    img_data = requests.get(image_url).content
    with open(input_file, 'wb') as handler:
        handler.write(img_data)

    response = runtime.invoke_endpoint(EndpointName='object-detection-2019-01-05-23-14-23-715',
                                      ContentType='image/jpeg',
                                      Body=img_data)
    # print(response)
    result = json.loads(response['Body'].read().decode())
    # print(result)

    object_categories = ['ship']
    # Setting a threshold 0.20 will only plot detection results that have a confidence score greater than 0.20.

    output_file = 'prediction.jpg'

    # Visualize the detections.
    encoded_image = visualize_detection(input_file, output_file, result['prediction'], object_categories, threshold)

    return render_template('output.html', result=encoded_image)

def visualize_detection(img_file, output_file, dets, classes=[], thresh=0.90):
    img=mpimg.imread(img_file)
    plt.imshow(img)
    height = img.shape[0]
    width = img.shape[1]
    colors = dict()
    for det in dets:
        (klass, score, x0, y0, x1, y1) = det
        if score < thresh:
            continue
        cls_id = int(klass)
        if cls_id not in colors:
            colors[cls_id] = (random.random(), random.random(), random.random())
        xmin = int(x0 * width)
        ymin = int(y0 * height)
        xmax = int(x1 * width)
        ymax = int(y1 * height)
        rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                             ymax - ymin, fill=False,
                             edgecolor=colors[cls_id],
                             linewidth=3.5)
        plt.gca().add_patch(rect)
        class_name = str(cls_id)
        if classes and len(classes) > cls_id:
            class_name = classes[cls_id]
        plt.gca().text(xmin, ymin - 2,
                        '{:s} {:.3f}'.format(class_name, score),
                        bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                                fontsize=12, color='white')
    # plt.imshow()
    plt.savefig(output_file)
    # im = Image.open(output_file)
    stream = io.BytesIO()
    # format = Image.registered_extensions()['.jpg']
    # im.save(stream,format)
    plt.savefig(stream)
    stream.seek(0)
    return base64.b64encode(stream.getvalue())

# include this for local dev

if __name__ == '__main__':
    app.run()
