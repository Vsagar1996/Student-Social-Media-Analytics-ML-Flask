## Student Social Media Addiction & Its Impact - End To End ML Project With Web Deployment (Flask · Docker · Hugging Face)

Live Demo:
👉 https://huggingface.co/spaces/Vsagar1996/Student_Social_Media-Impact_Analyzer

### Project Overview:

Social media plays a significant role in students’ daily lives, influencing their academic performance, mental well-being, and behavioral patterns. Excessive usage may lead to addiction-like symptoms, negatively affecting focus, productivity, and academic outcomes.

This project presents an end-to-end machine learning solution that analyzes student social media behavior and predicts:

 - Social media addiction score

 - Whether social media impacts academic performance

 - Whether a student is at high risk of social media addiction

The trained machine learning models are deployed as an interactive Flask web application, containerized using Docker, and hosted on Hugging Face Spaces, allowing real-time predictions through a user-friendly interface.

### Project Files:

- [Students Social Media Addiction.ipynb](Students%20Social%20Media%20Addiction.ipynb) – Jupyter Notebook containing the complete analysis

- [Students Social Media Addiction.csv](Students%20Social%20Media%20Addiction.csv) – Dataset used for the analysis

### Objectives:

Perform exploratory data analysis (EDA) on student social media usage data

Understand the relationship between social media usage, mental health, and academics

Build machine learning models for:

 - Addiction score prediction (Linear Regression)

 - Academic impact prediction (Logistic Regression)

 - High addiction risk prediction (Random Forest)

 - Apply preprocessing techniques such as scaling and one-hot encoding

 - Deploy trained models using Flask for real-time inference

 - Containerize the application using Docker

 - Host and demonstrate the project publicly using Hugging Face Spaces

### Machine Learning Models Used:

 - Addiction Score Prediction -	Linear Regression
 - Academic Performance Impact - Logistic Regression
 - High Addiction Risk - Random Forest Classifier

### Tools & Technologies Used:

 - Programming Language: Python

 - Libraries: Pandas, Scikit-learn, Matplotlib, Seaborn

 - Analysis workflow: Jupyter Notebook

 - Web Framework: Flask

 - Deployment: Docker, Hugging Face Spaces

 - Model Persistence: Pickle

 - Frontend: HTML & CSS (Flask Templates)

### Application Features:

 - User-friendly web interface for inputting student details

 - Dropdown-based categorical inputs generated from trained model features

 - Real-time predictions for all three ML models

 - Separate analysis pages for each prediction type

 - Deployed and publicly accessible live demo

### Future Work:

1. Use Standard Psychological Scales

   - Future studies can include validated psychological assessments such as PHQ-9 and GAD-7 to improve the reliability of mental health–related predictions.

2. Track Social Media Usage Over Time

   - Collecting daily or weekly usage data would help identify trends and early warning signs of increasing addiction using time-series analysis.

3. Improve Model Explainability and Performance

   - Further tuning of models and the use of explainability techniques can improve prediction accuracy and transparency.
