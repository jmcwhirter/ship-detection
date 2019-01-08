import os
import io
import boto3
import json
import random
import requests
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

# grab environment variables
runtime = boto3.client('runtime.sagemaker')

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
    plt.savefig(output_file)
    # plt.show()


def lambda_handler(event, context):
    print("Received event: " + json.dumps(event, indent=2))

    data = json.loads(json.dumps(event))
    image_url = data['image']

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
    threshold = data['threshold']
    output_file = 'prediction.jpg'

    # Visualize the detections.
    visualize_detection(input_file, output_file, result['prediction'], object_categories, threshold)

    im = Image.open(output_file)
    fp = io.BytesIO()
    format = Image.registered_extensions()['.jpg']
    im.save(fp,format)
    return fp.getvalue()

if __name__ == '__main__':
    payload = {'image': 'https://storage.googleapis.com/kaggle-media/competitions/Airbus/ships.jpg','threshold':0.40}
    response = lambda_handler(payload,'')
    print(response)
