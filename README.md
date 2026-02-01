## ML-Based Analysis of Student Social Media Addiction with Web Deployment (Flask · Docker · Hugging Face)

Live Demo:
👉 https://huggingface.co/spaces/Vsagar1996/Student_Social_Media-Impact_Analyzer

### Project Overview:

Social media plays a significant role in students’ daily lives, influencing their academic performance, mental well-being, and behavioral patterns. Excessive usage may lead to addiction-like symptoms, negatively affecting focus, productivity, and academic outcomes.

This project presents an end-to-end machine learning solution that analyzes student social media behavior and predicts:

 - Social media addiction score

 - Whether social media impacts academic performance

 - Whether a student is at high risk of social media addiction

The trained machine learning models are deployed as an interactive Flask web application, containerized using Docker, and hosted on Hugging Face Spaces, allowing real-time predictions through a user-friendly interface.

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

### Tech Stack:

 - Programming Language: Python

 - Libraries: Pandas, Scikit-learn, Matplotlib, Seaborn

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
