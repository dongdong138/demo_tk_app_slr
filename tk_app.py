import tkinter as tk
from tkinter import ttk, filedialog, simpledialog
from PIL import Image, ImageTk
import cv2
import mediapipe as mp
import numpy as np
import torch
from torchvision import transforms
from s3d_twostream import TwoStreamS3D
import pandas as pd

class SignLanguageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sign Language Detection App")

        self.model = TwoStreamS3D(num_classes=226).to('cpu')
        self.label_dict = pd.read_csv('SignList_ClassId_TR_EN.csv', index_col=['ClassId'])
        self.model.load_state_dict(torch.load('pretrained.pth', map_location=torch.device('cpu')))
        self.model.eval()

        self.create_widgets()

    def create_widgets(self):
        # Create video frame
        self.video_frame = ttk.Frame(self.root)
        self.video_frame.pack(padx=10, pady=10)

        # Create canvas for displaying video
        self.canvas = tk.Canvas(self.video_frame)
        self.canvas.pack()

        # Create buttons for camera selection and video file
        self.select_camera_button = ttk.Button(self.root, text="Select Camera", command=self.select_camera)
        self.select_camera_button.pack(pady=5)

        self.select_ip_camera_button = ttk.Button(self.root, text="Select IP Camera", command=self.select_ip_camera)
        self.select_ip_camera_button.pack(pady=5)

        self.select_video_button = ttk.Button(self.root, text="Select Video File", command=self.select_video_file)
        self.select_video_button.pack(pady=5)

        # Create start/stop button
        self.start_stop_button = ttk.Button(self.root, text="Start Detection", command=self.toggle_detection)
        self.start_stop_button.pack(pady=10)

        # Create label for displaying predictions
        self.prediction_label = ttk.Label(self.root, text="Prediction: ")
        self.prediction_label.pack(pady=5)

        # Set up Mediapipe holistic
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic()

        # Set up initial values
        self.is_detecting = False
        self.frame_list = []
        self.cap = None

        # Set up the GUI update
        self.update_gui()

    def toggle_detection(self):
        self.is_detecting = not self.is_detecting
        if self.is_detecting:
            self.start_stop_button["text"] = "Stop Detection"
        else:
            self.start_stop_button["text"] = "Start Detection"

    def select_camera(self):
        camera_index = simpledialog.askinteger("Select Camera", "Enter the camera index:")
        if camera_index is not None:
            self.cap = cv2.VideoCapture(camera_index)

    def select_ip_camera(self):
        ip_camera_url = simpledialog.askstring("Select IP Camera", "Enter the IP camera URL:")
        if ip_camera_url is not None:
            self.cap = cv2.VideoCapture(ip_camera_url)

    def select_video_file(self):
        file_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video files", "*.mp4;*.avi")])
        if file_path:
            self.cap = cv2.VideoCapture(file_path)

    def update_gui(self):
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                if self.is_detecting:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.holistic.process(rgb_frame)
                    if results.pose_landmarks:
                        if len(self.frame_list) < 120:
                            self.frame_list.append(frame.copy())
                        else:
                            self.frame_list[-1] = frame.copy()
                            num_frame = 16
                            step = len(self.frame_list) // num_frame
                            self.frame_list = self.frame_list[::step][:16]
                            sign = self.detect_sign(self.frame_list)
                            prediction_text = self.label_dict.iloc[sign[0].item()].to_string(index=False)
                            self.update_prediction_label(prediction_text)
                    else:
                        self.frame_list = []

                # Display the frame on the canvas
                frame = cv2.resize(frame, (640, 480))
                self.display_frame(frame)

        # Schedule the next update
        self.root.after(10, self.update_gui)

    def detect_sign(self, frames):
        poses = self.get_poses(frames)
        frames, poses = self.transform_frames(frames), self.transform_frames(poses)
        outputs = self.model(frames, poses)
        _, predict = torch.max(outputs, 1)
        return predict

    def get_poses(self, frames):
        mp_face_detection = mp.solutions.face_detection
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose

        poses = []

        with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection, \
                mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_detection:

            for frame in frames:
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                w, h, c = image_rgb.shape
                image_black = np.zeros((w, h, c), dtype=np.uint8)

                face_results = face_detection.process(image_rgb)
                if face_results.detections:
                    for detection in face_results.detections:
                        mp_drawing.draw_detection(frame, detection)

                pose_results = pose_detection.process(image_rgb)
                if pose_results.pose_landmarks:
                    mp_drawing.draw_landmarks(image_black, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                    connections = [
                        (mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_SHOULDER),
                        (mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.RIGHT_SHOULDER)]

                    for connection in connections:
                        start_point = pose_results.pose_landmarks.landmark[connection[0]]
                        end_point = pose_results.pose_landmarks.landmark[connection[1]]
                        start_x, start_y = int(start_point.x * frame.shape[1]), int(start_point.y * frame.shape[0])
                        end_x, end_y = int(end_point.x * frame.shape[1]), int(end_point.y * frame.shape[0])
                        cv2.line(image_black, (start_x, start_y), (end_x, end_y), (255, 255, 255), 2)

                poses.append(image_black)
        return poses

    def get_transform(self):
        return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def transform_frames(self, frames):
        transform = self.get_transform()
        frames = [transform(frame) for frame in frames]
        frames = torch.stack(frames)
        frames = frames.reshape((frames.shape[1], frames.shape[0], frames.shape[2], frames.shape[3]))
        return frames.unsqueeze(0)

    def display_frame(self, frame):
        # Convert the frame to RGB format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert the frame to ImageTk format
        img = Image.fromarray(rgb_frame)
        img = ImageTk.PhotoImage(image=img)

        # Update the canvas with the new frame
        self.canvas.config(width=img.width(), height=img.height())
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
        self.canvas.image = img

    def update_prediction_label(self, prediction_text):
        tmp = prediction_text.split('\n')
        prediction_text = tmp[0].strip() + ' ' + tmp[1].strip()

        # Increase font size
        font_size = 16  # Adjust the font size as needed
        font_style = ("Helvetica", font_size, "bold")

        self.prediction_label.config(text=prediction_text, foreground="green", font=font_style)

if __name__ == "__main__":
    root = tk.Tk()
    app = SignLanguageApp(root)
    root.mainloop()
