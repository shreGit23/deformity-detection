import io
import streamlit as st
import backend
from PIL import Image
import shutil
import os
import time

st.title("Test Streamlit App")
uploaded_file = st.file_uploader("Please upload an image", type=["jpg", "png", "jpeg"])

# st.sidebar.title("Model Selection")

# all_models = (
#     'Kidney Stone Detection',
#     'Fracture Detection',
#     'Pneumonia Classification'
#     )

# model_option = st.sidebar.selectbox('Select an analysis type', all_models)

if uploaded_file is not None:

    image = Image.open(io.BytesIO(uploaded_file.read()))
    st.image(image, caption="Uploaded Image", use_column_width=True)
    var_1, var_2, var_3, var_4 = backend.model_prediction(image)
    
    if var_1 is not None:
        if 'Detection' in var_4:
            prediction, version, model, model_used = var_1, var_2, var_3, var_4

            if version == 'v5':
                exp_dir = 'runs/detect/exp'
                if os.path.exists(exp_dir) and os.listdir(exp_dir):
                    image_name = os.listdir(exp_dir)
                    image_prediction = os.path.join(exp_dir, image_name[0])

                    st.image(image_prediction, caption="Processed Image", use_column_width=True)

                    for info in prediction.xyxy[0]:
                        x1, y1, x2, y2, confidence, class_id = info
                        st.write(f"**Detected:** {model.names[int(class_id)]} -> **Confidence:** {confidence * 100:.2f}%")
                        print(f"Detected: {model.names[int(class_id)]} -> Confidence: {confidence * 100:.2f}%")

                    shutil.rmtree('runs')
                else:
                    st.write("No prediction results found for YOLOv5.")
                    print("No prediction results found for YOLOv5.")

            elif version == 'v8':

                for info in prediction:
                    predict_dir = info.save_dir

                if os.path.exists(predict_dir) and os.listdir(predict_dir):
                    image_name = os.listdir(predict_dir)
                    image_prediction = os.path.join(predict_dir, image_name[0])

                    st.image(image_prediction, caption="Processed Image", use_column_width=True)
                    time.sleep(1)
                    shutil.rmtree('runs')
                
                else:
                    st.write("No prediction results found for YOLOv8.")
                    print("No prediction results found for YOLOv8.")

        elif 'Classification' in var_4:
            class_name, confidence_score, model, model_option = var_1, var_2, var_3, var_4

            st.write(f'**Class Name:** {class_name}')
            st.write(f'**Confidence Score:** {confidence_score*100:.2f}%')

        else:
            print("BROKEN CODE LOL!!!!!")
    else:
        st.write("**If this is a medical image, we may currently lack a detection or classification system for this type of image.**")
        print("Error : Model Not Found For This Image")