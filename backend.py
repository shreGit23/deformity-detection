# IMPORT REQUIRED LIBRARIES
import torch
import tensorflow as tf
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageOps
import os
from ultralytics import YOLO 
import shutil
import tempfile
import streamlit as st

def get_model_path(model): # add your models file path (.pt) for yolo and (.h5) for classification models.
    model = model.strip()
    model_paths = {
        'Kidney Stone Detection': [r'models\Kidney_Stone_Model (81.3% mAP)\weights\best.pt', 'yolov5'],
        'Fracture Detection': [r'models\Fracture_Localization_Model (74% mAP)\weights\best.pt', 'yolov5'],
        'Brain Tumor Detection': [r'models\Brain Tumor Detection Model\best.pt', 'yolov8'],
        'Pneumonia Classification': [r'models\Pneumonia Classification Model\pneumonia_classifier.h5', 'tensorflow'],
        'alzheimer classification': [r'D:\python programs\Internship\image_diagnosis\env\streamlit_v3\models\alzheimer classification\Alzheimer_Incetionv3_model.h5','tensorflow']
    }
    return model_paths.get(model)



def load_model(selected_model):  # function to load required model.
    model_data = get_model_path(selected_model)

    model_path, model_type = model_data

    if model_data is None:
        print(f"Model not found. Please check the model_paths in backend.py at line 8 and model_option in app.py at line 15.")
        return None

    print(f"Model Found: {selected_model}\nPath: {model_path}\nModel Type: {model_type}\n\n")

    if model_type == 'yolov5':
        return torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
    
    elif model_type == 'yolov8':
        return YOLO(model_path)
    
    elif model_type == 'tensorflow':
        return tf.keras.models.load_model(model_path, compile=False)
    else:
        print("Please add model_paths or its type if you haven't specified in backend.py, line 8")
        return None



def yolo(image, model, version):
    if version == 'v5':
        prediction = model(image)
        prediction.save()
        return prediction, version
    
    elif version == 'v8':
        image_name = os.path.splitext(os.path.basename(image))[0]
        save_dir = os.path.join('yolov8_predictions', image_name)
        os.makedirs(save_dir, exist_ok=True)  

        prediction = model(source=image, save=True, save_dir=save_dir, name=image_name)
        
        shutil.rmtree('yolov8_predictions')
        return prediction, version



def classification_model(image_file, model, model_option):
    
    model_data = get_model_path(model_option)
    model_path, _ = model_data

    labels_path = os.path.join(os.path.dirname(model_path), 'labels.txt')
    if not os.path.exists(labels_path):
        print(f"Labels file not found at {labels_path}")
        return None
    
    class_names = open(labels_path, "r").readlines()
    size = (224, 224)
    image = ImageOps.fit(image_file, size, Image.LANCZOS).convert('RGB')
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    predictions = model.predict(data)
    index = np.argmax(predictions)
    class_name = class_names[index].strip()
    confidence_score = predictions[0][index]
    
    print(f'Class Name: {class_name}\nConfidence Score: {confidence_score*100:.2f}%')
    return class_name, confidence_score



def check_model_requirement(image):
    print("\n__________________________________________\nPredicting Model Required For The Image...........")
    model_path = r'D:\python programs\Internship\image_diagnosis\env\streamlit_v3\models\Model_Selecter\keras_model.h5'
    model = tf.keras.models.load_model(model_path, compile=False)
    
    labels_path = r'D:\python programs\Internship\image_diagnosis\env\streamlit_v3\models\Model_Selecter\labels.txt'
    if not os.path.exists(labels_path):
        print(f"Labels file not found at {labels_path}")
        return None
    
    class_names = open(labels_path, "r").readlines()
    size = (224, 224)
    
    image = ImageOps.fit(image, size, Image.LANCZOS).convert('RGB')
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    
    predictions = model.predict(data)
    index = np.argmax(predictions)
    class_name = class_names[index].strip()
    print(f'Model Requirement Predicted: {class_name}\n')
    return class_name



def model_prediction(image): 
    model_option = check_model_requirement(image).strip()
    
    if model_option == 'Non-Medical Image':
        st.write(f"**Sorry, this is not a medical image. Please try again with a medical image**")
        return None, None, None, None 
    
    else:
        model = load_model(model_option)

        if not model:
            return None, None, None, None

        yolov5_models = ['Kidney Stone Detection', 'Fracture Detection']
        yolov8_models = ['Brain Tumor Detection']
        tensorflow_models = ['Pneumonia Classification', 'alzheimer classification']

        if model_option in yolov5_models:
            print("-- In model_prediction it's on YOLOv5\n")
            prediction, version = yolo(image, model, 'v5')
            return prediction, version, model, model_option

        elif model_option in yolov8_models:
            print("-- In model_prediction it's on YOLOv8\n")
            
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_image_file:
                temp_image_path = temp_image_file.name
                if image.mode == 'RGBA':
                    image = image.convert('RGB')
                image.save(temp_image_path)
            
            prediction, version = yolo(temp_image_path, model, 'v8')
            os.unlink(temp_image_path)

            return prediction, version, model, model_option

        elif model_option in tensorflow_models:
            print("-- In model_prediction it's on Tensorflow\n")
            class_name, confidence_score = classification_model(image, model, model_option)
            return class_name, confidence_score, model, model_option

        else:
            print(f"Sorry , we currently don't have Detection or Classification model for this image")
            return None, None, None, None
