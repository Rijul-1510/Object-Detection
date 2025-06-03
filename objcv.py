# from ultralytics import YOLO
# import cv2
import os

# img = cv2.imread(r"C:\Users\Admin\hellopython\ObjectRecognition\Obj\img.jpg", cv2.IMREAD_COLOR)
# model = YOLO(r"C:\Users\Admin\hellopython\ObjectRecognition\Obj\best.pt")
# results = model(img)  # return a list of Results objects
# # print(results)
# # Process results list
# for result in results:
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs
#     obb = result.obb  # Oriented boxes object for OBB outputs
#     result.show()  # display to screen
#     result.save(filename="result.jpg")  # save to disk 
# # cv2.imshow("image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

BASE_PATH = os.getcwd()
print(BASE_PATH)