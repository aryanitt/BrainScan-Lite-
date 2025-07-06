# 🧠 BrainScan Lite – MBTI Personality Predictor

BrainScan Lite is a web-based machine learning application that predicts a user's Myers-Briggs Type Indicator (MBTI) personality type based on their responses to a short quiz. The application uses a trained ML classification model (Decision Tree, KNN, etc.) and provides real-time personality prediction through a beautiful and responsive web interface built with Flask and Bootstrap.

---

## 🚀 Features

- 🔢 Quiz of 10 Likert-scale questions (1 to 5)
- 🧠 Predicts one of the 16 MBTI types (e.g., INFP, ESTJ)
- 📊 ML-backed model using scikit-learn pipelines
- 🖥️ Real-time prediction via a Flask web server
- 💾 Model preprocessing with scaling and encoding
- 🌐 Responsive and modern UI with Bootstrap 5

---

## 🧰 Tech Stack

| Category           | Tools/Libraries                            |
|--------------------|---------------------------------------------|
| Backend            | Python, Flask                              |
| Machine Learning   | scikit-learn, pandas, numpy                |
| Frontend           | HTML, CSS, Bootstrap                       |
| Preprocessing      | LabelEncoder, StandardScaler, Pipelines    |
| Deployment Ready   | Folder structure for production pipelines  |

---

## 📁 Project Structure

BrainScan Lite/
├── artifacts/ # Saved preprocessor, label encoder, and model
├── notebook/ # Dataset (CSV) and experiment notebooks
├── templates/ # HTML templates for Flask
│ ├── index.html
│ └── home.html
├── static/ # (Optional) custom CSS/images if any
├── src/
│ ├── components/ # Ingestion, transformation, trainer
│ ├── pipeline/ # Prediction pipeline
│ ├── utils.py
│ ├── exception_config.py
│ ├── logger_config.py
├── app.py # Main Flask application
├── requirements.txt # Python dependencies
└── README.md # Project documentation"# BrainScan-Lite-" 
