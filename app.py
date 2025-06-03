from flask import Flask, jsonify, render_template, request
from ultralytics import YOLO
import os
import cv2

app = Flask(__name__)

model = YOLO(r"ObjectRecognition/best.pt")

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
            img = cv2.imread(path_save, cv2.IMREAD_COLOR)
            results = model.predict(img)  # return a list of Results objects
            # BASE_PATH = os.path.dirname(__file__)
            PREDICTION_PATH = os.path.join(BASE_PATH, r"static\prediction",upload_file.filename)
            for result in results:
                boxes = result.boxes  # Boxes object for bounding box outputs
                masks = result.masks  # Masks object for segmentation masks outputs
                keypoints = result.keypoints  # Keypoints object for pose outputs
                probs = result.probs  # Probs object for classification outputs
                obb = result.obb  # Oriented boxes object for OBB outputs
                # result.show()  # display to screen
                result.save(PREDICTION_PATH)  # save to disk
            upload_URL = modify_url(UPLOAD_PATH)
            prediction_URL = modify_url(PREDICTION_PATH)
            # print("THE UPLOAD URL IS", upload_URL)

            return render_template('results.html',upload_URL=upload_URL,prediction_URL=prediction_URL)


# def prediction(path):
#     # Read image


#     return results

if __name__ == '__main__':
    app.run(debug=True)
