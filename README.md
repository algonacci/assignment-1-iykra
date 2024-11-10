# Generative AI Capstone Project: Deepfake Detection for e-KYC

This project focuses on developing a Generative AI solution to detect deepfake images, specifically tailored for electronic Know Your Customer (e-KYC) processes. The solution enhances security in identity verification by identifying AI-generated image manipulations that could be used for fraudulent impersonation.

## 1. Generative AI Use Case

### Problem Statement

This project addresses the detection of deepfake images to improve security in e-KYC processes. Deepfake content poses a significant risk in identity verification, as it allows individuals to impersonate others using AI-generated image manipulations. The objective is to develop a model that can detect deepfake patterns in image frames to ensure secure and reliable e-KYC procedures.

### Reason for Choosing this Use Case

With the rise in identity fraud and misuse of deepfake technology, particularly in digital transactions and e-KYC, a robust detection system is essential. Implementing deepfake detection enhances trust and security in digital identity verification, making it a critical application of generative AI in the finance and legal compliance sectors.

## 2. Source System

- **Data Source**: This project utilizes the CelebA dataset for pre-training on facial feature detection, with the FaceForensics++ dataset used for deepfake-specific training.
- **Links to Source**:
  - [CelebA Dataset on Kaggle](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)
  - [FaceForensics++ Dataset](https://github.com/ondyari/FaceForensics)

## 3. Target System

- **Target System**: Data is stored locally within structured directories (`data/celeba/img_align_celeba` for CelebA).
- **Reason for Choice**: Local storage allows fast and efficient access during training, which is essential for managing high-resolution images and video frames. A local setup facilitates faster testing and model iteration, ensuring more rapid development and validation.

## 4. Model or Technique

- **Model**: This project employs a combination of Convolutional Neural Networks (CNNs) for image analysis, along with a Discriminator model tailored to detect features associated with deepfakes.
- **Deepfake Detection Model**: A CNN-based model (inspired by GAN Discriminators) is adapted to classify images or video frames as real or fake based on texture, pixel consistency, and other patterns typical of deepfake artifacts.
- **Technique**: The model is trained with adversarial techniques, leveraging GANs to generate deepfakes, which are then used to train the detection model. This setup ensures that the model learns both from authentic images and AI-generated counterfeits, creating a balanced and robust classifier for e-KYC validation.

## 5. End-to-End Data Pipeline Architecture

The following are the key steps in the data pipeline:

1. **Data Ingestion**: Download the CelebA and FaceForensics++ datasets from Kaggle or other sources (`data_ingestion.py`).
2. **ETL (Extract, Transform, Load)**: Preprocess images and video frames by resizing, normalizing, and applying frame extraction on videos if necessary (`etl.py`).
3. **Modeling**: Initialize the deepfake detection model, a CNN-based classifier, along with potential GAN models for generating training data (`modelling.py`).
4. **Training**: Train the model using real and deepfake data, focusing on minimizing classification errors (`training.py`).
5. **Deployment**: Deploy the trained model to identify deepfake attempts in real-time during e-KYC processes (`app.py`).

## Deployment

The deployed model is accessible at:

- [Cloud Run URL](https://assignment-1-iykra-902243842780.asia-southeast1.run.app)
