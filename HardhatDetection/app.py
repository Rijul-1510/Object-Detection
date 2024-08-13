from flask import Flask, jsonify, render_template, request, Response
from ultralytics import YOLO
import os
import cv2
import supervision as sv
import dlib
import matplotlib.pyplot as plt
import cv2
import numpy as np

app = Flask(__name__)

model = YOLO(r"best_v10_50.pt")
class_names = ['helmet', 'head', 'person']  # Replace with actual class names of your model


@app.route('/')
def index():
    return render_template('index.html')

def modify_url(url):
    url = url.replace('\\', '/')
    static_index = url.find('static')
    if static_index != -1:
        url = url[static_index + len('static'):]
        url = url.strip('/')
    return url

def detect_objects(image, results):
    # Extract bounding boxes, confidence scores, and class IDs
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes
    scores = results[0].boxes.conf.cpu().numpy()  # Confidence scores
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)  # Class IDs

    # Define colors for bounding boxes
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

    # Draw bounding boxes and labels on the image
    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = map(int, box)  # Ensure coordinates are integers
        label = f'{class_names[class_id]}: {score:.2f}'
        color = colors[class_id % len(colors)]  # Cycle through the colors

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image, (boxes, scores, class_ids)


@app.route('/showres', methods=['GET', 'POST'])
def application():
    if request.method == 'POST':
        if 'image_name' in request.files:  
            upload_file = request.files['image_name']
            BASE_PATH = os.path.dirname(__file__)
            UPLOAD_PATH = os.path.join(BASE_PATH, r"static\upload",upload_file.filename)
            # print("Upload Folder: ", UPLOAD_PATH )
            path_save = os.path.join(UPLOAD_PATH)
            # print("THE UPLOADED FILE IS",upload_file.filename)
            # Store image in upload directory
            upload_file.save(UPLOAD_PATH)
            # Take image make preds
            image = cv2.imread(path_save, cv2.IMREAD_COLOR)
            detector = dlib.get_frontal_face_detector()
            results = model(image, conf=0.7)

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            if not faces:
                face_detected = False
            else:
                face_detected = True

            newimage=image.copy()
            for face in faces:
                x, y, w, h = (face.left(), face.top(), face.width(), face.height())
                newimage= cv2.rectangle(newimage, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            
            FACE_DETECTION_PATH = os.path.join(BASE_PATH, r"static\prediction\face",upload_file.filename)
            cv2.imwrite(FACE_DETECTION_PATH,newimage)


# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            detected_image, (boxes, scores, class_ids) = detect_objects(image, results)

            # Filter detections based on class names
            selected_classes = [class_names.index('helmet')]  # Only select the 'helmet' class
            filtered_indices = [i for i, class_id in enumerate(class_ids) if class_id in selected_classes]

            filtered_boxes = boxes[filtered_indices]
            filtered_scores = scores[filtered_indices]
            filtered_class_ids = class_ids[filtered_indices]

            # Count detections for helmets
            helmet_count = len(filtered_class_ids)

            helmet_detected= helmet_count>0

            # Save and display the result
            resized_image = cv2.resize(detected_image, (256, 400))

            # cv2.imshow(resized_image)

            print("HELMET ",helmet_detected)
            print("FACE ", face_detected)

            hmessage=''
            fmessage=''

            if helmet_detected == False:
                hmessage="Please ensure HELMET is visible"
            if face_detected == False:
                fmessage="Please ensure FACE is visible"
            if helmet_detected and face_detected:
                print("Helmet Validation - PASSED")
                validation = 'PASS'
            else:
                print("Helmet Validation - FAIL")
                validation = 'FAILED'

            # Extract bounding boxes, confidences, and class IDs
            HELMET_DETECTION_PATH = os.path.join(BASE_PATH, r"static\prediction\helmet",upload_file.filename)
            for result in results:
                boxes = result.boxes  # Boxes object for bounding box outputs
                masks = result.masks  # Masks object for segmentation masks outputs
                keypoints = result.keypoints  # Keypoints object for pose outputs
                probs = result.probs  # Probs object for classification outputs
                obb = result.obb  # Oriented boxes object for OBB outputs
                # result.show()  # display to screen
                result.save(HELMET_DETECTION_PATH)  # save to disk
            upload_URL = modify_url(UPLOAD_PATH)
            hat_URL = modify_url(HELMET_DETECTION_PATH)
            face_URL = modify_url(FACE_DETECTION_PATH)
            # print("THE UPLOAD URL IS", upload_URL)



            return render_template('index.html', upload_URL=upload_URL, hat_URL=hat_URL, face_URL=face_URL, validation=validation, hmessage=hmessage, fmessage=fmessage )


if __name__ == '__main__':
    app.run(debug=True)
