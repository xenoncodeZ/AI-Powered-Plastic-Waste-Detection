AI Powered Plastic Waste Detection
This project leverages computer vision and AI to detect plastic waste in various environments, aiming to contribute to environmental monitoring and cleanup efforts. The core of the system utilizes a custom-trained YOLOv5 model for real-time detection and a separate script for image recognition and map generation.

Features
Real-time Plastic Waste Detection: Utilizes a YOLOv5 model for live detection of plastic waste, suitable for integration with cameras or video streams.

Image Upload Recognition: Allows users to upload images for analysis, identifying plastic waste within them.

Map Generation: Generates a visual representation (e.g., a simple map) of detected plastic waste locations, potentially aiding in cleanup planning.

Technologies Used
YOLOv5: For object detection (real-time and image-based).

Gradio: For creating a user-friendly web interface for real-time detection.

Python: The primary programming language for all scripts.

OpenCV (likely): For image processing tasks.

Other libraries (e.g., NumPy, Matplotlib): For data handling and potential visualization.

Project Structure
yolov5_gradiotrash.py: Script responsible for the real-time plastic waste detection using YOLOv5 and a Gradio interface.

trash_detection.py: Script handling image upload recognition and the generation of maps based on detection data.

README.md: This file.

(Additional files like requirements.txt, model weights, sample data, etc., would typically be here)

Setup and Installation
To get this project up and running, follow these steps:

Clone the repository:

git clone <your-repository-url>
cd "AI Powered Plastic Detection"

(Replace <your-repository-url> with the actual URL of your Git repository.)

Create a virtual environment (recommended):

python -m venv venv
.\venv\Scripts\activate   # On Windows
source venv/bin/activate # On macOS/Linux

Install dependencies:

pip install -r requirements.txt

(You will need a requirements.txt file listing all Python dependencies like torch, ultralytics, gradio, opencv-python, etc. If you don't have one, you'll need to install them individually, e.g., pip install torch torchvision opencv-python ultralytics gradio)

Download YOLOv5 weights:

You'll need the pre-trained YOLOv5 weights (e.g., yolov5s.pt) and your custom-trained weights for plastic detection. Place them in the appropriate directory (e.g., a weights/ folder).

Usage
Real-time Detection (yolov5_gradiotrash.py)
To run the real-time detection interface:

python yolov5_gradiotrash.py

This will typically launch a Gradio web interface in your browser, where you can use your webcam or upload a video for live detection.

Image Upload Recognition & Map Generation (trash_detection.py)
To use the image upload and map generation features:

python trash_detection.py

This script will likely have its own interface or command-line arguments for uploading images and processing them to generate detection data and maps. Refer to the script's internal documentation or arguments for specific usage.

Model Limitations
It's important to note that the current AI model has been trained on a dataset of approximately 5000 images. Due to the relatively small dataset size, the accuracy of the detection model is currently a little low. This may result in:

False positives: Detecting non-plastic items as plastic.

False negatives: Failing to detect actual plastic waste.

Reduced performance in diverse environments or with varying lighting conditions.

Future Enhancements
Dataset Expansion: Significantly increase the training dataset size with more diverse images of plastic waste.

Model Fine-tuning: Explore advanced fine-tuning techniques or different YOLOv5 architectures.

Improved Map Visualization: Enhance the map generation with more interactive features and detailed location data.

Deployment: Consider deploying the application for easier access and wider use.

License
This project is open-source and available under the MIT License. (Consider adding a LICENSE file to your repository if you haven't already.)
