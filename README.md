# Breast Cancer Prediction Using ANN

## 📌 About the Project
This project uses an Artificial Neural Network (ANN) to classify breast tumors as **Benign or Malignant** using the Breast Cancer Wisconsin Diagnostic dataset.

It includes data preprocessing, Exploratory Data Analysis (EDA), model training, evaluation, and a Streamlit web application for interactive predictions.

## 🛠️ Technologies Used
- Python
- NumPy and Pandas
- Matplotlib and Seaborn
- Scikit-learn
- TensorFlow and Keras
- Streamlit
- Jupyter Notebook

## ⚙️ Project Workflow
1. Load and explore the dataset.
2. Clean and preprocess the data.
3. Visualize feature distributions and correlations.
4. Split the dataset into training and testing sets.
5. Scale features using StandardScaler.
6. Build and train an ANN model.
7. Evaluate performance using accuracy, precision, recall, F1-score, and ROC-AUC.
8. Save the trained model and scaler.
9. Build an interactive Streamlit application.

## 🧠 ANN Architecture
- Input layer: 30 features
- Hidden layer 1: 64 neurons with ReLU activation
- Dropout: 20%
- Hidden layer 2: 32 neurons with ReLU activation
- Output layer: 1 neuron with sigmoid activation

**Optimizer:** Adam  
**Loss Function:** Binary Cross-Entropy  
**Early Stopping:** Used to help reduce overfitting.

## 🚀 How to Run

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal, usually `http://localhost:8501`.

## 📁 Project Files
- `ANN.ipynb` — Data analysis, preprocessing, training, and evaluation.
- `app.py` — Streamlit web application.
- `data.csv` — Dataset.
- `breast_cancer_ann.keras` — Saved ANN model.
- `scaler.pkl` — Saved feature scaler.
- `requirements.txt` — Project dependencies.

## 🎯 Learning Outcomes
- Data preprocessing and visualization.
- Artificial Neural Network implementation.
- Binary classification and model evaluation.
- Deploying a machine learning model using Streamlit.

## ⚠️ Disclaimer
This project is for educational purposes only. Its predictions are not a medical diagnosis and should not be used for clinical decisions.

## 👨‍💻 Author
**Tharnish Y**  
B.Tech — Artificial Intelligence and Data Science
