import dlib
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import supervision as sv
import cv2
import numpy as np


model = YOLO(r"best_v10_50.pt")


# Define class names
class_names = ['helmet', 'head', 'person']  # Replace with actual class names of your model


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

# Load the image
image_path = r'C:\Users\30024\Desktop\hellopython\HardhatDetection\side_helmet.jpg'
image = cv2.imread(image_path)

detector = dlib.get_frontal_face_detector()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = detector(gray)
if not faces:
  face_detected = False
else:
  face_detected = True

# Perform inference using your model (adjust the function call as needed)
results = model(image, conf=0.7)

# Detect objects in the image
detected_image, (boxes, scores, class_ids) = detect_objects(image, results)

# Filter detections based on class names
selected_classes = [class_names.index('helmet')]  # Only select the 'helmet' class
filtered_indices = [i for i, class_id in enumerate(class_ids) if class_id in selected_classes]

filtered_boxes = boxes[filtered_indices]
filtered_scores = scores[filtered_indices]
filtered_class_ids = class_ids[filtered_indices]

# Count detections for helmets
helmet_count = len(filtered_class_ids)

# Print the number of helmets detected
# print(f'Number of helmets: {helmet_count}')

helmet_detected= helmet_count>0

# Save and display the result
output_path = "result.jpg"
cv2.imwrite(output_path, detected_image)
resized_image = cv2.resize(detected_image, (256, 400))

# cv2.imshow(resized_image)

print("HELMET ",helmet_detected)
print("FACE ", face_detected)

if helmet_detected and face_detected:
  print("Helmet Validation PASS")
else:
  print("Helmet Validation FAIL")



# import cv2
# import dlib


# image_path = r'C:\Users\30024\Desktop\hellopython\HardhatDetection\no_helmet_2.jpg'
# image = cv2.imread(image_path)

# detector = dlib.get_frontal_face_detector()
# img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# img =img.astype('uint8')


# faces = detector(img)

# if not faces:
#   face_detected = False
# else:
#   face_detected = True

# print(face_detected)
