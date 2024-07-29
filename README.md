# deformity-detection
Deformity detection in medical images using machine learning and computer vision techniques.

# Deformity Detection in Medical Images

## Project Overview

This project focuses on detecting and localizing deformities in medical images using machine learning and computer vision techniques. The system is designed to handle multiple types of medical images and identify various conditions, including brain tumors, fractures, kidney stones, and pneumonia. The models are deployed on Streamlit for easy accessibility and interaction.


## Intallation

Create a Python environment and install the necessary libraries for running Streamlit. For doing so, download the requirements.txt file and then run it on the terminal of your environment.

Example:

pip install -r requirements.txt

## Technologies Used

TensorFlow: machine learning framework.

Torch: Deep learning framework.

Streamlit: Web application framework for deploying models.

Ultralytics: For YOLO models.

SciPy: Scientific computations.

Scikit-learn: Machine learning library.

Pillow: Image processing library.

Matplotlib: Image visualization.

OpenCV: Image processing.

Pandas: Data manipulation and analysis.

NumPy: Numerical computations.

Seaborn: Statistical data visualization.

PyYAML: YAML file handling.

Psutil: System and process utilities.


## Project Functionality

The project includes a model selector with five labels:

1. Non-medical

2. Brain Tumor

3. Fracture

4. Kidney

5. Pneumonia

Each label directs the input image to the respective model for deformity detection. The models are trained and stored as.h5 files, which are deployed on Streamlit for real-time detection and visualization.

## Deployment

The application is deployed using Streamlit, making it user-friendly and accessible for testing and demonstration purposes.

### Files Attached

.h5 model files for each condition.

Source codes for preprocessing, enhancement, segmentation, and deployment.

This setup ensures that the project is comprehensive, covering various medical conditions and providing a streamlined workflow from image preprocessing to deformity detection and visualization.
