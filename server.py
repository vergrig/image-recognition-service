import json

import torch
from flask import Flask, request, jsonify
from prometheus_flask_exporter.multiprocess import GunicornInternalPrometheusMetrics
from prometheus_client import Counter

from furl import furl
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models

import requests


app = Flask(__name__, static_url_path="")
metrics = GunicornInternalPrometheusMetrics(app)
PREDICTION_COUNT = Counter("predictions_total", "Number of predictions", ["label"])

THRESHOLD = 0.75
model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

transform_pipeline = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

def prepare(img_link):
    image = Image.open(requests.get(img_link, stream=True).raw)
    img_data = transform_pipeline(image).unsqueeze(0)
    data = img_data.tolist()
    return data


@app.route("/predict", methods=['POST'])
@metrics.gauge("api_in_progress", "requests in progress")
@metrics.counter("api_invocations_total", "number of invocations")
@metrics.counter("app_http_inference_count", "app_http_inference_count_total")

def predict():

    img_path = request.get_json(force=True)
    data = {'data': prepare(img_path['url'])}

    features = torch.tensor(data['data'])

    result = model(features)
    detection_num = len(result[0]['labels'])

    ans_list = [COCO_INSTANCE_CATEGORY_NAMES[result[0]['labels'][i]] 
                for i in range(detection_num)
                if result[0]['scores'][i].item() > THRESHOLD]

    for label in ans_list:
        PREDICTION_COUNT.labels(label=label).inc()

    return jsonify({
        "objects": ans_list
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

