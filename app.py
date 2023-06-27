import gradio as gr
import tensorflow as tf
import cv2
import numpy as np

new_model = tf.keras.models.load_model('saved_model/main_bilstm_model')


def predict_video(video_file):
    SEQUENCE_LENGTH = 16  # Set your desired sequence length
    CLASSES_LIST = ["NonViolence","Violence"]  # Replace with your own class labels
    IMAGE_HEIGHT = 64  # Set your desired image height
    IMAGE_WIDTH = 64  # Set your desired image width

    video_reader = cv2.VideoCapture(video_file)
    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_list = []
    predicted_class_name = ''
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames_window = max(int(video_frames_count / SEQUENCE_LENGTH), 1)

    for frame_counter in range(SEQUENCE_LENGTH):
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        ret, frame = video_reader.read()

        if not ret:
            break

        # Prediction
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
        normalized_frame = resized_frame / 255
        frames_list.append(normalized_frame)

    # Assuming you have `new_model` and `CLASSES_LIST` defined elsewhere
    predicted_labels_probabilities = new_model.predict(np.expand_dims(frames_list, axis=0))[0]
    predicted_label = np.argmax(predicted_labels_probabilities)
    predicted_class_name = CLASSES_LIST[predicted_label]
    confidence = predicted_labels_probabilities[predicted_label]
    return f'Predicted: {predicted_class_name}\nConfidence: {confidence}'

inputs = gr.inputs.Video(label="Upload a video file")
outputs = gr.outputs.Textbox()

iface = gr.Interface(fn=predict_video, inputs=inputs, outputs=outputs, title='Video Classification')
iface.launch()