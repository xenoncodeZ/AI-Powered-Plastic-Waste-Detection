import torch
import numpy as np
import cv2
from PIL import Image
import folium
import gradio as gr

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

# Real-time detection function
def detect_realtime(video_source=0):  # Default video_source=0 for webcam
    cap = cv2.VideoCapture(video_source)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame)
        boxes = results.xyxy[0].cpu().numpy()
        plastic_count = 0
        total_count = len(boxes)
        
        for box in boxes:
            x1, y1, x2, y2, conf, cls_id = box
            cls_name = all_labels[int(cls_id)]
            color = (0, 255, 0) if cls_name in plastic_labels else (255, 0, 0)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f"{cls_name} {conf:.2f}", (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            if cls_name in plastic_labels:
                plastic_count += 1
        
        cv2.imshow('Real-Time Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Quit with 'q' key
            break

    cap.release()
    cv2.destroyAllWindows()

# Gradio interface
def process_input(image, toggle):
    if image:
        processed_img, result_text, detections = detect_trash(image)
        map_message = generate_map(detections)
        return processed_img, result_text, map_message
    return None, "Please upload an image.", None

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
    title="Trash Detection & Real-Time Mapping",
    description="Analyze images for trash content and visualize detections dynamically on a map.",
)

interface.launch()