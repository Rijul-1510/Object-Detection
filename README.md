# **AI VALIDATION OF PERSONAL PROTECTIVE EQUIPMENT**

AI Validation of Personal Protective Equipment (PPE) is a machine learning based system designed to enhance safety compliance at industrial and telecom sites. By leveraging the real-time capabilities [...]

# **Key Features**
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

## **System Architecture**

![System Architecture](static/images/system_architecture.png)

## **Installation**

```bash
pip install -r requirements.txt
python app.py
```

## **Usage**
1. Start the Flask server
   ```bash
   $ python app.py
   ```
2. Navigate to `http://localhost:5000` in a browser.
3. Upload an image and press Submit.
4. View Input Image, Output Image with bounding boxes, and Data Retrieved in the results page.

## **Result and Performance**

![Sample Result](static/images/sample_result.png)
![Performance Chart](static/images/performance_chart.png)

