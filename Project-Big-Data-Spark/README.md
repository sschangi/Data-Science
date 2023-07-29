The churn problem refers to the phenomenon where customers or users of a service, such as a mobile app or a subscription-based platform, stop using the service and move on to other alternatives. Sparkify app is a music streaming service (similar to platforms like Spotify or Apple Music). In the context of the Sparkify app, churn would refer to users who stop using the app and cancel their subscriptions or disengage from the platform. In this scenario, Churn can be a critical issue for several reasons like Revenue Loss, User Acquisition Costs, Customer Lifetime Value, and Market Reputation. In a competitive market, customer retention is crucial for maintaining a competitive edge. If churn is not addressed effectively, rival apps may attract Sparkify's users to their platforms.

In this project, we classify the users of the Sparkify app into two groups: churned and non-churned users. The focus of this endeavor lies in analyzing extensive data using Pyspark and developing a web application tailored for churn classification.

In the project, we try to predict the churn rates of a music app, called Sparkify. The full dataset is 12GB, of which we analyze a mini subset of 128MB. 

## List of all the files in the repository:
- `Sparkify.ipynb`: contains all steps: Loading and Cleaning data, EDA analysis, Feature Engineering, ML modeling, and Evaluations.
- `app\run.py`: module to deploy and run the flask app into a local server.
- `app\templates\master.html`: display the main page of the web app.
- `app\templates\expanalysis.html`: display the EDA analysis.
- `app\templates\featureengineering.html`: display plots for the relavent features.
- `app\templates\mlmodel.html`: display the results of the best ML models, i.e, Logistic regression (LR), and Random Forest (RF) on the training and the test data along with coefficient importance plot for the LR model.

## Intruction to run the Web Application
- Run the following command in the app directory and start a local web app using the flask server:
  ```
  - python run.py
  - Go to http://localhost:3001/
  ```

## Libraries used
Create the conda environment with:

`conda create --name myenv python=3.10`

List of used dependencies are as follow:

- pandas=1.5.3
- numpy=1.23.5
- scikit-learn=1.2.1
- matplotlib=3.7.1
- Flask=0.12.5
- plotly=5.11.0
- pyspark=3.4.1

## Acknowledgements
I would like to thank [Udacity](https://www.udacity.com/) for providing the Sparkify dataset.

