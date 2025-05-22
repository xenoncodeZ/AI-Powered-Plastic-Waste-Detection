import torch
import numpy as np
import cv2
from PIL import Image
import gradio as gr
import folium  # Library for creating maps

# Load the pre-trained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)
plastic_labels = ['plastic_bottle', 'plastic_bag', 'plastic_wrapper']
all_labels = model.names

# Image-based detection function
def detect_trash(image):
    results = model(image)
    boxes = results.xyxy[0].cpu().numpy()
    output_img = np.array(image).copy()
    plastic_count = 0
    total_count = len(boxes)

    detections = []  # Store detection data

    for box in boxes:
        x1, y1, x2, y2, conf, cls_id = box
        cls_name = all_labels[int(cls_id)]
        color = (0, 255, 0) if cls_name in plastic_labels else (255, 0, 0)

        if cls_name in plastic_labels:
            plastic_count += 1

        cv2.rectangle(output_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(output_img, f"{cls_name} {conf:.2f}", (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add detection info (assuming geo-coordinates are provided)
        detections.append({
            "label": cls_name,
            "confidence": float(conf),
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
        })

    plastic_percent = (plastic_count / total_count) * 100 if total_count > 0 else 0

    return Image.fromarray(output_img), f"Plastic Content: {plastic_percent:.2f}%", detections

# Map visualization function
def generate_map(detections):
    # Create a Folium map object centered on a default location (latitude, longitude)
    map_object = folium.Map(location=[26.5, 85.5], zoom_start=12)

    for detection in detections:
        # Example: Place a marker for each detection with the label and confidence
        # Replace the coordinates with actual detection geo-coordinates if available
        folium.Marker(
            location=[26.5, 85.5],  # Example coordinates
            popup=f"{detection['label']} ({detection['confidence']:.2f})",
            icon=folium.Icon(color="red" if detection["label"] in plastic_labels else "blue")
        ).add_to(map_object)

    # Save map as an HTML file
    map_object.save("map.html")
    return "Map generated successfully! Open map.html to view."

# Gradio interface with map and detection functionalities
def process_input(image, toggle):
    if image:
        processed_img, result_text, detections = detect_trash(image)
        map_message = generate_map(detections)
        return processed_img, result_text, map_message
    else:
        return None, "Please upload an image.", None

# Custom CSS for better UI
custom_css = """
body {
    background: linear-gradient(135deg, #1e1e2f, #3a3a5f);
    font-family: 'Arial', sans-serif;
    color: #d3d3d3;
}
h1 {
    color: #8ab4f8;
    font-size: 36px;
    text-align: center;
    margin-bottom: 20px;
}
.gr-button {
    background-color: #8ab4f8;
    color: #ffffff;
    border-radius: 8px;
    padding: 10px;
    margin: 10px;
    width: 200px;
    text-align: center;
}
.gr-button:hover {
    background-color: #4174a8;
    transition: 0.3s;
}
.gr-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 100vh;
}
"""

# Gradio interface
interface = gr.Interface(
    fn=process_input,
    inputs=[
        gr.Image(type="pil", label="Upload Your Image"),
        gr.Checkbox(label="Enable Map Visualization"),
    ],
    outputs=[
        gr.Image(type="pil", label="Processed Image"),
        gr.Textbox(label="Results"),
        gr.Textbox(label="Map Status"),
    ],
    title="Trash Detection & Mapping",
    description="Analyze images for plastic content, visualize detections on a map!",
    css=custom_css
)

# Launch the Gradio interface
interface.launch()