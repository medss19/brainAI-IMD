# brainAI - Brain Tumor Analysis and Prediction

**brainAI** is an advanced web application developed to analyze, classify, and predict brain tumors. By leveraging cutting-edge technologies like data augmentation with Generative Adversarial Networks (GANs), ResNet for tumor type classification, and time series analysis for tumor growth predictions, BrainAI helps in early tumor detection and personalized treatment planning.

Link to dataset for GANs and detection/classification - https://zenodo.org/records/12735702/files/brain-tumor-mri-dataset.zip?download=1

## Key Features

### 1. Tumor Type Classification
- **Data Augmentation with GAN**: Enhances the dataset by generating synthetic tumor images for different tumor types, improving model performance and generalization.
- **ResNet Architecture**: Utilizes the powerful ResNet model for accurate classification of various types of brain tumors based on MRI scans.

### 2. Tumor Growth Analysis
- **Time Series Analysis**: Tracks and analyzes the progression of brain tumors over time, providing insights into the growth patterns and stages of tumor development.

### 3. Early Tumor Prediction
- **Predictive Analysis**: Identifies potential early-stage tumors based on MRI data, enabling proactive treatment and improving patient outcomes.

## Technologies Used

- **GAN (Generative Adversarial Networks)**: Used for data augmentation to create more diverse tumor images.
- **ResNet (Residual Networks)**: A deep learning architecture for classifying tumor types from MRI scans.
- **Time Series Analysis**: For analyzing tumor growth over time, offering predictive insights.
- **Python**: Core programming language for implementing machine learning models and algorithms.
- **TensorFlow/Keras**: For building and training the neural networks used in the project.
- **Flask/Django**: Backend web framework for handling the web application and API.
- **JavaScript (React/Angular)**: Frontend for the interactive user interface.

## How It Works

1. **Data Augmentation**: The system uses GAN to generate additional MRI images for each tumor type to ensure a robust training set.
   
2. **Tumor Classification**: MRI scans are fed into a ResNet model, which classifies the type of brain tumor (e.g., glioma, meningioma, pituitary).

3. **Growth Prediction**: Time series analysis helps in forecasting the growth rate of tumors based on historical data.

4. **Early Detection**: The system identifies early signs of potential tumors, providing users with predictive analysis for timely diagnosis.

