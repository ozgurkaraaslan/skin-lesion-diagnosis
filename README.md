# Skin Lesion Classification and Web-Based Diagnostic System

## Project Overview

Early detection of skin lesions, such as melanoma, is a critical pattern recognition challenge in dermatology. This project aims to automate the classification process of dermatoscopic images using deep learning and machine learning techniques to assist medical professionals in early diagnosis.

Utilizing the **HAM10000 dataset**, which contains over 10,000 images categorized into 7 different disease classes, the system extracts features and trains models (including CNNs, SVM, K-NN, and NNs) to achieve high accuracy. The final trained model is integrated into a **FastAPI backend** and served through a user-friendly **web interface**, with the entire architecture containerized using **Docker** for seamless deployment.

## Team Members & Roles

* **Halil İbrahim BAYAT:** Feature extraction design, SVM/K-NN/NN implementation, mathematical formulation, literature review, backend development, Docker deployment.
* **Büşra Doran:** Dataset preparation & preprocessing, feature extraction pipeline, hyperparameter tuning, performance evaluation.
* **Özgür Karaaslan:** Implementation support, model training & testing, experiment execution, web interface design, cross-validation.

## Project Structure

* `data/`: Contains the HAM10000 dataset (Ignored by Git; must be downloaded locally).
* `notebooks/`: Jupyter notebooks for exploratory data analysis (EDA) and experimental model training.
* `src/`: Source code including the FastAPI backend, machine learning scripts, and web interface files.
* `models/`: Saved weights of the trained models.
* `docs/`: Project documentation, final reports, and evaluation metrics (e.g., confusion matrix).

## Installation & Usage

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Halilbyt/skin-lesion-diagnosis.git
   cd skin-lesion-diagnosis
   ```

2. **Build the Docker image:**

   ```bash
   docker compose build
   ```

3. **Run the Docker container:**

   ```bash
   docker compose up
   ```

4. **Access the web interface:**
   Open your browser and navigate to `http://localhost:8000` to input data and receive diagnostic predictions.
