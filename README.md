# Iris Flower Classification 🌸

## Table of Content:
...

***

## 1. Project Overview 

## 2. The Iris Flower Dataset 

### A. Dataset Source 

### B. Data Exploration and Data Visualization 

## 2. Machine Learning Model 

## 3. Flask Web App

## 4. Files Structure
```
├── app #Website folder
│   ├── app.py #Responsible of running the website
│   └── templates
│       ├── index.html # Allows the user to input and predict new flower properties 
│   └── Static 
│       ├── index.css # This file has the Cascading Style Sheets of the index.html
|
├── data
│   ├── dataset.csv # The Iris flower dataset
│   ├── dataset.db #The prepared dataset as SQLite database
│   └── process_data.py #Responsible for dataset preparation
|
├── models
│   ├── model.pkl #The Logistic Regression Model
│   └── train_classifier.py #Responsible for creating the machine learning model
|
├── images #This folder contains all images for the readme file
│   ├── flower.jpg
|
└── README.md #Readme file 
```

## 5. Requirments
In order to to run this project, you must have [Python3](https://www.python.org/) installed on your machine. You also must have all listed libraries inside the `requirments.txt` so run the following command to install them: 
```
pip3 install -r requirments.txt
```

## 6. Running Process 
This secions explains how to run each part of this project using the command prompt or terminal

### A. Process Data
 Data Exploration and Data Visualization have been done in `process_data.ipynb` file which is inside the data folder

### B. Training the classifier
To re-train the classifier, you must go inside the `models` directory using the terminal or the command prompt and run the following:
```shell
python3 train_classifier.py ../data/<database_name>.db <model_name>.pkl
```

### C. Run the Flask Web App
To run the web app, you must go inside the `app` directory using the terminal or the command prompt and run the following:
```shell
python3 run.py
```
The link of the website will be `0.0.0.0:3001`

## 7. Conclusion
...

## 8. Acknowledgements
I would like to express my appreciation to [Misk Academy](https://misk.org.sa/en/) and [Udacity](https://www.udacity.com/) for the amazing work on the data science course and the support they give us to build this project
