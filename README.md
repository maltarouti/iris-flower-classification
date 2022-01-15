# Iris Flower Classification 🌸

## Table of Content:
* [1. Project Overview](project_overview)
* [2. The Iris Flower Dataset](the_iris_flower_dataset)
    * [A. Dataset Source](dataset_source)
    * [B. Data Exploration and Data Visualization](data_exploration_and_data_visualization)
* [3. Machine Learning Model](machine_learning_model)
* [4. Flask Web App](flask_web_app)
* [5. Files Structure](files_structure)
* [6. Requirments](requirments)
* [7. Running Process](running_process)
    * [A. Process Data](process_data)
    * [B. Training the classifier](training_the_classifier)
    * [C. Run the Flask Web App](run_the_flask_web_app)
* [8. Conclusion](conclusion)
* [9. Acknowledgements](acknowledgements)

***
<a id=project_overview></a>
## 1. Project Overview 
In this project, we will analyze the iris flower dataset, which has three species: Setosa, Versicolor and Virginica. Each flower class has around 50 records in the dataset. The main goal of this project is to create a classification model that uses the length and width measurements of the sepal and petal to categorize new flowers.

![image](https://github.com/Murtada-Altarouti/Iris-flower-classification/blob/main/images/flower.jpg)

<a id=the_iris_flower_dataset></a>
## 2. The Iris Flower Dataset 

<a id=dataset_source></a>
### A. Dataset Source 
The Iris flower dataset was taken from [Kaggle](https://www.kaggle.com/arshid/iris-flower-dataset) as a comma-separated values (CSV), and it contains a set of 150 records under 5 attributes - Petal Length, Petal Width, Sepal Length, Sepal width and Class(Species).

<a id=data_exploration_and_data_visualization></a>
### B. Data Exploration and Data Visualization
The data exploration and data visualization were done inside the `/data/process_data.ipynb`, but here are some of the findings:

![image](https://github.com/Murtada-Altarouti/Iris-flower-classification/blob/main/images/dataset.png)

As seen above, there are almost 50 records of each flower class in the dataset

![image](https://github.com/Murtada-Altarouti/Iris-flower-classification/blob/main/images/values.png)

As it shown above, the sepal range is between 4.3cm and 7.9cm in length and 2.0cm and 4.4cm in width. But the petal range is between 1.0cm and 6.9cm in length and 0.1cm and 2.5cm in width.

The chart also shows that Virginica has the longest sepal length which may reach 7.9cm, as opposed to Setosa, which has a range of 4.3cm to 5.8cm. On the other hand, Setosa has the widest sepals at 4.4cm and Virginica has the highest petal length and width.

<a id=machine_learning_model></a>
## 3. Machine Learning Model
The machine learning model was trained on the Iris flower dataset using The [scikit learn](https://scikit-learn.org/stable/) Python library. The model is [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), which is an excellent classifier since it applies the one-vs-rest principle to this multi-class situation.

<a id=flask_web_app></a>
## 4. Flask Web App
The Flask Web App allows the user to use the trained model to make predictions on new flowers and find their species easily
![image](https://github.com/Murtada-Altarouti/Iris-flower-classification/blob/main/images/webapp.png)

<a id=files_structure></a>
## 5. Files Structure
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

<a id=requirments></a>
## 6. Requirments
In order to to run this project, you must have [Python3](https://www.python.org/) installed on your machine. You also must have all listed libraries inside the `requirments.txt` so run the following command to install them: 
```
pip3 install -r requirments.txt
```
<a id=running_process></a>
## 7. Running Process 
This secions explains how to run each part of this project using the command prompt or terminal

<a id=process_data></a>
### A. Process Data
To look at the data exploration and data visualization, please open `/data/process_data.ipynb` with [Jupyter Notebook](https://jupyter.org/).

<a id=training_the_classifier></a>
### B. Training the classifier
To re-train the classifier, you must go inside the `models` directory using the terminal or the command prompt and run the following:
```shell
python3 train_classifier.py ../data/<database_name>.db <model_name>.pkl
```
<a id=run_the_flask_web_app></a>
### C. Run the Flask Web App
To run the web app, you must go inside the `app` directory using the terminal or the command prompt and run the following:
```shell
python3 run.py
```
The link of the website will be `0.0.0.0:3001`

<a id=conclusion></a>
## 8. Conclusion
We conclude that it may be difficult for non-experts to identify iris flower species, machine learning makes it much easier to find out the flower class with 94 percent accuracy. In the future, we can try to develop a deep learning model utilizing neural networks, which may yield even better and more accurate results.

<a id=acknowledgements></a>
## 9. Acknowledgements
I would like to express my appreciation to [Misk Academy](https://misk.org.sa/en/) and [Udacity](https://www.udacity.com/) for the amazing work on the data science course and the support they give us to build this project
