# AI VALIDATION OF PERSONAL PROTECTIVE EQUIPMENT
```bash
AI Validation of Personal Protective Equipment (PPE) is a machine learning based system designed to enhance safety compliance at industrial and telecom sites.
By leveraging the real-time capabilities of the YOLOv8 model, this solution automatically detects whether essential safety gear—such as helmets, jackets, gloves, and shoes—is worn correctly by workers.
The system features an intuitive web interface built with Flask, enabling users to upload images and receive immediate visual and textual feedback.
This project addresses a critical need for automating PPE compliance monitoring, aiming to reduce workplace accidents and improve operational efficiency across hazardous environments.
```

# Key Features
```bash
Real-Time Object Detection 
Detects helmets, jackets, gloves, and shoes using YOLOv8 with high-speed inference.

Compliance Validation Logic
Automatically determines if all required PPE items are present—only marks a worker as compliant when all gear is correctly worn.

Web-Based Interface (Flask)
User-friendly image upload page with a results dashboard showing input image, annotated output, and textual detection summary.

Lightweight and Easy to Deploy
Minimal dependencies; runs locally with Python and Flask—ideal for quick prototyping or integration into larger safety systems.

Modular Architecture
Easy to extend with new object classes, logic, or deployment targets (e.g., edge devices, video streams).
```
## **Directory Structure** 

```python
Object-Detection/
├── app.py                 # Flask application
├── best.pt                # Trained YOLOv8 weights
├── requirements.txt
├── static/
│   ├── upload/            # Uploaded images
│   └── prediction/        # Images with bounding boxes
├── templates/
│   ├── index.html         # Upload page
│   └── results.html       # Results dashboard
└── README.md
```

# System Architecture 

![Screenshot 2025-06-03 111409](https://github.com/user-attachments/assets/ba72a631-a732-4424-9222-3b06a926e94b)


# Installation

```bash
pip install -r requirements.txt
python app.py
```

# Usage
1. Start the Flask server
   ```bash
   $ python app.py
   ```
2. Navigate to `http://localhost:5000` in a browser.
3. Upload an image and press Submit.
4. View Input Image, Output Image with bounding boxes, and Data Retrieved in the results page.

# Result and Performance
# **Object Detection Results**
The model, built on YOLOv8, demonstrates high accuracy in detecting key personal protective equipment (PPE) items across diverse environments. The inference results clearly identify:
Helmets
Jackets
Shoes

Each detection is tagged with a confidence score, with many predictions exceeding 0.9, indicating strong model confidence. The model performs robustly in varied conditions including different backgrounds, lighting, angles, and poses. Both face-visible and face-occluded scenarios are handled accurately.

![Screenshot 2024-11-21 114407](https://github.com/user-attachments/assets/de538e10-c242-41a6-aa65-327491f03503)

# **Performance Metrics**
The Precision–Confidence Curve illustrates the precision of the model across different confidence thresholds.
Mean precision for all classes peaks at 1.00 at a confidence of 0.884, showing the model is well-calibrated.

![Screenshot 2024-11-21 111304](https://github.com/user-attachments/assets/46cd8413-8345-47f9-b570-dc3947ac84af)

# **Overall Precision:**

Excellent performance with precision ≥ 0.9 across most classes.
Model confidently separates true positives from false positives, making it reliable for real-time compliance validation.

# Final Dashboard

![Picture1](https://github.com/user-attachments/assets/6992ae85-49fc-4f23-a239-5a5edf6ffa1b)
