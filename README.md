# Data-Science
This repo contains the codes and documentations for the Udacity Data Science nanodegree projects. 

## Project: Write a Data Scientist Blog Post
In this project, we aim to **write a Data Scientist Blog Post** analyzing the data set for [Seattle Airbnb Data](https://www.kaggle.com/datasets/airbnb/seattle).

We particularly answer the following questions:
- How much Airbnb homes are earning in certain time frames and areas?
- What are the properties helping to predict price?
- Can we find the listings with positive and negative reviews?

For a summary of the analysis see the post [How to choose an airbnb place in Seattle?](https://medium.com/@schangiz2002/how-to-choose-an-airbnb-place-in-seattle-556e04dba571)

## Project: Disaster Response Project
In this Project, we use a data set containing real messages that were sent during disaster events. By creating a machine learning pipeline, we categorize these events so that one can send the messages to an appropriate disaster relief agency.

The project includes a web app where an emergency worker can input a new message and get classification results in several categories. The files and scripts are located in the [Project-Disaster-Response-Pipeline](https://github.com/sschangi/Data-Science/tree/main/Project-Disaster-Response-Pipeline) folder.

## Project: Recommendations with IBM
In this project, we analyze the interactions that users have with articles on the [IBM Watson Studio platform](https://dataplatform.cloud.ibm.com/login), and make recommendations to them about new articles we think they will like, using rank-based, collaborative filtering, and svd decomposition approaches in recommendation systems. The files and scripts are located in the [Project-Recommendation-Engine](https://github.com/sschangi/Data-Science/tree/main/Project-Recommendation-Engine) folder.

## Project: Big Data with Spark
The churn problem refers to the phenomenon where customers or users of a service, such as a mobile app or a subscription-based platform, stop using the service and move on to other alternatives. Sparkify app is a music streaming service (similar to platforms like Spotify or Apple Music). In the context of the Sparkify app, churn would refer to users who stop using the app and cancel their subscriptions or disengage from the platform. In this scenario, Churn can be a critical issue for several reasons like Revenue Loss, User Acquisition Costs, Customer Lifetime Value, and Market Reputation. In a competitive market, customer retention is crucial for maintaining a competitive edge. If churn is not addressed effectively, rival apps may attract Sparkify's users to their platforms.

In this project, we classify the users of the Sparkify app into two groups: churned and non-churned users. The focus of this endeavor lies in analyzing extensive data using Pyspark and developing a web application tailored for churn classification.

The full dataset is 12GB, of which we analyze a mini subset of 128MB. The project includes a web app using Flask. The app can run locally by running `python run.py` and going to `http://localhost:3001/`. For the complete codes and documentation see [Project-Big-Data-Spark](https://github.com/sschangi/Data-Science/tree/main/Project-Big-Data-Spark) folder. 

## Dependencies
Create the conda environment with:

`conda create --name myenv python=3.10`

List of used dependencies are as follow:

- pandas=1.5.3
- numpy=1.23.5
- scikit-learn=1.2.1
- scipy=1.10.0
- matplotlib=3.7.1
- seaborn=0.12.2
- vaderSentiment=3.3.2
- Flask=0.12.5
- plotly=5.11.0
- pyspark=3.4.1
