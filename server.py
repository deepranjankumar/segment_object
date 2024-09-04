from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import cvlib as cv
# from cvlib.object_detection import draw_bbox
import json
import base64
import os
import torch
import supervision as sv
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from collections import OrderedDict


app = Flask(__name__)

def detect_common_objects(image):
    # Detect common objects using cvlib
    
    boxes, labels, scores = cv.detect_common_objects(image)
    new_labels = list(OrderedDict.fromkeys(labels))  # Use OrderedDict to preserve the order of labels
    boxes_based_on_label = OrderedDict((label, []) for label in new_labels)  # Initialize OrderedDict to store boxes based on label
    
    # Store boxes based on label in the OrderedDict
    for idx, label in enumerate(labels):
        boxes_based_on_label[label].append(boxes[idx])
    print(boxes_based_on_label)
    objects = [{'label': label.upper(), 'count': len(boxes_based_on_label[label]), 'boxes': boxes_based_on_label[label]} for label in new_labels]
    return objects


def generate_segmented_image(image_rgb, box):
    # Load the pretrained model
    HOME = os.getcwd()
    
    CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    MODEL_TYPE = "vit_h"
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
    mask_predictor = SamPredictor(sam)
    
    # Set up the mask predictor and predict masks
    mask_predictor.set_image(image_rgb)
    box = np.array(box)
    masks, _, _ = mask_predictor.predict(box=box, multimask_output=True)

    # Create a blank canvas with the same size as the original image
    foreground_only = np.zeros_like(image_rgb)

    # Overlay segmented parts onto the blank canvas
    for mask in masks:
        foreground_only[mask > 0] = image_rgb[mask > 0]

    # Convert the segmented image to base64
    _, buffer = cv2.imencode('.jpg', foreground_only)
    segmented_image_encoded = base64.b64encode(buffer).decode('utf-8')

    return segmented_image_encoded

def draw_bbox(output_image, boxes, labels, scores):
    # Use OrderedDict to preserve the order of labels
    new_labels = list(OrderedDict.fromkeys(labels))
    boxes_based_on_label = OrderedDict((label, []) for label in new_labels)  

    # Store boxes based on label in the OrderedDict
    for idx, label in enumerate(labels):
        boxes_based_on_label[label].append(boxes[idx])
    print(boxes_based_on_label)
    # Draw bounding boxes and labels
    for label, box_list in boxes_based_on_label.items():
        for box in box_list:
            # Extract box coordinates
            x1, y1, x2, y2 = box
            # Draw bounding box
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Get class name from the label
            class_name = label.split("_")[0]
            # Draw label with index
            cv2.putText(output_image, f"{class_name}{box_list.index(box)}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    return output_image



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    nparr = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    objects = detect_common_objects(img)
    box, labels, scores = cv.detect_common_objects(img)
    
    # Draw bounding boxes and labels
    output_image = np.copy(img)

    output_image = draw_bbox(output_image, box, labels, scores)

    # Convert the output image to base64
    _, buffer = cv2.imencode('.jpg', output_image)
    img_encoded = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'objects': objects, 'processed_image': img_encoded})

@app.route('/segment', methods=['POST'])
def segment():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part'})

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected image'})

    # Read the image file
    nparr = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Get the box coordinates
    box_data = request.form.get('box')  # Accessing form data instead of request body
    box = [int(coord) for coord in json.loads(box_data)]

    # Generate segmented image
    segmented_image_encoded = generate_segmented_image(img, box)

    return jsonify({'segmented_image': segmented_image_encoded})


if __name__ == '__main__':
    app.run(debug=True)