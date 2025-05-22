# import torch
# import numpy as np
# import cv2
# from PIL import Image
# import gradio as gr
 
# # Load trained modelmodel = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)
 
# # Define plastic classes (adjust based on your dataset)
# plastic_labels = ['plastic_bottle', 'plastic_bag', 'plastic_wrapper']
# all_labels = model.names
 
# def detect_trash(image):
#     results = model(image)
#     boxes = results.xyxy[0].cpu().numpy()
 
#     output_img = np.array(image).copy()
#     plastic_count = 0
#     total_count = len(boxes)
 
#     for box in boxes:
#         x1, y1, x2, y2, conf, cls_id = box
#         cls_name = all_labels[int(cls_id)]
#         color = (0, 255, 0) if cls_name in plastic_labels else (255, 0, 0)
 
#         if cls_name in plastic_labels:
#             plastic_count += 1
 
#         cv2.rectangle(output_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
#         cv2.putText(output_img, f"{cls_name} {conf:.2f}", (int(x1), int(y1)-10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
 
#     plastic_percent = (plastic_count / total_count) * 100 if total_count > 0 else 0
#     return Image.fromarray(output_img), f"Plastic Content: {plastic_percent:.2f}%"
 
# interface = gr.Interface(
#     fn=detect_trash,
#     inputs=gr.Image(type="pil"),
#     outputs=[gr.Image(type="pil"), gr.Textbox()],
#     title="pLittre",
#     description="Upload an image and save the world from plastic🌍."
# )
 
# interface.launch()
# import torch
# import numpy as np
# import cv2
# from PIL import Image
# import gradio as gr

# # Load the pre-trained model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)
# plastic_labels = ['plastic_bottle', 'plastic_bag', 'plastic_wrapper']
# all_labels = model.names

# # Function for image-based detection
# def detect_trash(image):
#     results = model(image)
#     boxes = results.xyxy[0].cpu().numpy()
#     output_img = np.array(image).copy()
#     plastic_count = 0
#     total_count = len(boxes)

#     for box in boxes:
#         x1, y1, x2, y2, conf, cls_id = box
#         cls_name = all_labels[int(cls_id)]
#         color = (0, 255, 0) if cls_name in plastic_labels else (255, 0, 0)

#         if cls_name in plastic_labels:
#             plastic_count += 1

#         cv2.rectangle(output_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
#         cv2.putText(output_img, f"{cls_name} {conf:.2f}", (int(x1), int(y1)-10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#     plastic_percent = (plastic_count / total_count) * 100 if total_count > 0 else 0
#     return Image.fromarray(output_img), f"Plastic Content: {plastic_percent:.2f}%"

# # Function for real-time video-based detection
# def detect_realtime(toggle):
#     if not toggle:
#         return None, "Real-time detection is not active."
#     cap = cv2.VideoCapture(0)  # Open default webcam

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
#         results = model(frame)
#         boxes = results.xyxy[0].cpu().numpy()
#         output_frame = frame.copy()

#         for box in boxes:
#             x1, y1, x2, y2, conf, cls_id = box
#             cls_name = all_labels[int(cls_id)]
#             color = (0, 255, 0) if cls_name in plastic_labels else (255, 0, 0)
#             cv2.rectangle(output_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
#             cv2.putText(output_frame, f"{cls_name} {conf:.2f}", (int(x1), int(y1)-10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#         cv2.imshow("Real-time Detection - Caffeine CoderZ", cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR))

#         if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to stop
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     return None, "Real-time detection stopped."

# # Custom CSS for layout and styling
# custom_css = """
# body {
#     background: linear-gradient(135deg, #1e1e2f, #3a3a5f);
#     font-family: 'Arial', sans-serif;
#     color: #d3d3d3;
# }
# h1 {
#     color: #8ab4f8;
#     font-size: 36px;
#     text-align: center;
#     margin-bottom: 20px;
# }
# .gr-button {
#     background-color: #8ab4f8;
#     color: #ffffff;
#     border-radius: 8px;
#     padding: 10px;
#     margin: 10px;
#     width: 200px;
#     text-align: center;
# }
# .gr-button:hover {
#     background-color: #4174a8;
#     transition: 0.3s;
# }
# .gr-container {
#     display: flex;
#     flex-direction: column;
#     align-items: center;
#     justify-content: center;
#     min-height: 100vh;
# }
# """

# # Gradio interface with both functionalities
# interface = gr.Interface(
#     fn=lambda img, toggle: detect_realtime(toggle) if toggle else detect_trash(img),
#     inputs=[
#         gr.Image(type="pil", label="Upload Your Image"),
#         gr.Checkbox(label="Enable Real-time Detection (Webcam)"),
#     ],
#     outputs=[
#         gr.Image(type="pil", label="Processed Image"),
#         gr.Textbox(label="Results"),
#     ],
#     title="Caffeine CoderZ",
#     description="Analyze images for plastic content or toggle real-time detection for live video!",
#     css=custom_css
# )

# interface.launch()

import torch
import numpy as np
import cv2
from PIL import Image
import gradio as gr
 
# Load the pre-trained model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)
plastic_labels = ['plastic_bottle', 'plastic_bag', 'plastic_wrapper']
all_labels = model.names
 
# Function for image-based detection
def detect_trash(image):
    results = model(image)
    boxes = results.xyxy[0].cpu().numpy()
    output_img = np.array(image).copy()
    plastic_count = 0
    total_count = len(boxes)
 
    for box in boxes:
        x1, y1, x2, y2, conf, cls_id = box
        cls_name = all_labels[int(cls_id)]
        color = (0, 255, 0) if cls_name in plastic_labels else (255, 0, 0)
 
        if cls_name in plastic_labels:
            plastic_count += 1
 
        cv2.rectangle(output_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(output_img, f"{cls_name} {conf:.2f}", (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
 
    plastic_percent = (plastic_count / total_count) * 100 if total_count > 0 else 0
    return Image.fromarray(output_img), f"Plastic Content: {plastic_percent:.2f}%"
 
# Function for real-time video-based detection
def detect_realtime(toggle):
    if not toggle:
        return None, "Real-time detection is not active."
    cap = cv2.VideoCapture(0)  # Open default webcam
 
    while True:
        ret, frame = cap.read()
        if not ret:
            break
 
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
        results = model(frame)
        boxes = results.xyxy[0].cpu().numpy()
        output_frame = frame.copy()
 
        for box in boxes:
            x1, y1, x2, y2, conf, cls_id = box
            cls_name = all_labels[int(cls_id)]
            color = (0, 255, 0) if cls_name in plastic_labels else (255, 0, 0)
            cv2.rectangle(output_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(output_frame, f"{cls_name} {conf:.2f}", (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
 
        cv2.imshow("Real-time Detection - Caffeine CoderZ", cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR))
 
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to stop
            break
 
    cap.release()
    cv2.destroyAllWindows()
    return None, "Real-time detection stopped."
 
# Custom CSS for layout and styling
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
 
# Gradio interface with both functionalities
interface = gr.Interface(
    fn=lambda img, toggle: detect_realtime(toggle) if toggle else detect_trash(img),
    inputs=[
        gr.Image(type="pil", label="Upload Your Image"),
        gr.Checkbox(label="Enable Real-time Detection (Webcam)"),
    ],
    outputs=[
        gr.Image(type="pil", label="Processed Image"),
        gr.Textbox(label="Results"),
    ],
    title="Caffeine CoderZ",
    description="Analyze images for plastic content or toggle real-time detection for live video!",
    css=custom_css
)
 
interface.launch()
 
import torch
import numpy as np
import cv2
from PIL import Image
import gradio as gr
 
# Load the pre-trained model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)
plastic_labels = ['plastic_bottle', 'plastic_bag', 'plastic_wrapper']
all_labels = model.names
 
# Image-based detection
def detect_trash(image):
    results = model(image)
    boxes = results.xyxy[0].cpu().numpy()
    output_img = np.array(image).copy()
    plastic_count = 0
    total_count = len(boxes)
 
    for box in boxes:
        x1, y1, x2, y2, conf, cls_id = box
        cls_name = all_labels[int(cls_id)]
        color = (0, 255, 0) if cls_name in plastic_labels else (255, 0, 0)
        if cls_name in plastic_labels:
            plastic_count += 1
        cv2.rectangle(output_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(output_img, f"{cls_name} {conf:.2f}", (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
 
    plastic_percent = (plastic_count / total_count) * 100 if total_count > 0 else 0
    return Image.fromarray(output_img), f"Plastic Content: {plastic_percent:.2f}%"
 
# Webcam snapshot-based detection
def detect_from_webcam():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
 
    if not ret:
        return None, "Failed to capture image from webcam."
 
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame)
    boxes = results.xyxy[0].cpu().numpy()
    output_img = frame.copy()
    plastic_count = 0
    total_count = len(boxes)
 
    for box in boxes:
        x1, y1, x2, y2, conf, cls_id = box
        cls_name = all_labels[int(cls_id)]
        color = (0, 255, 0) if cls_name in plastic_labels else (255, 0, 0)
        if cls_name in plastic_labels:
            plastic_count += 1
        cv2.rectangle(output_img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(output_img, f"{cls_name} {conf:.2f}", (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
 
    plastic_percent = (plastic_count / total_count) * 100 if total_count > 0 else 0
    return Image.fromarray(output_img), f"Plastic Content: {plastic_percent:.2f}%"
 
# UI logic based on toggle
def process_input(image, toggle):
    if toggle:
        return detect_from_webcam()
    else:
        return detect_trash(image)
 
# Custom CSS
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
        gr.Checkbox(label="Use Webcam Snapshot Instead")
    ],
    outputs=[
        gr.Image(type="pil", label="Processed Image"),
        gr.Textbox(label="Results")
    ],
    title="Caffeine CoderZ",
    description="Analyze images for plastic content or take a webcam snapshot for instant detection!",
    css=custom_css
)
 
interface.launch()
 