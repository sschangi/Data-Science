# import libraries
import re
import json
import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import plotly
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
from flask import Flask
from flask import render_template, request, jsonify

app = Flask(__name__)

# index webpage displays cool visuals and receives user input text for model
@app.route("/")
@app.route("/index")
def index():
    """Generate Plots in the HTML index page.
    
        Args:
            None
            
        Returns:
            render_template(render_template): Render template for the plots  
        
    """

    df = pd.read_json("data/mini_sparkify_event_data.json", lines=True)

    # Count of null entries in each column
    null_counts = df.isnull().sum()
    null_counts = null_counts[null_counts > 0].sort_values(ascending=False)

    # Count of empty entries in each column (assuming empty means an empty string '')
    empty_counts = (df == '').sum()
    empty_counts = empty_counts[empty_counts > 0].sort_values(ascending=False)

    # Creating DataFrames from the counts
    null_df = pd.DataFrame(null_counts, columns=['Null_Count'])
    empty_df = pd.DataFrame(empty_counts, columns=['Empty_Count'])

    graphs = [
        plot(null_df.T, 'Null'), 
        plot(empty_df.T, 'Empty-Value')
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template("master.html", ids=ids, graphJSON=graphJSON)

@app.route("/expanalysis")
def expanalysis():
    """Generate Explatory Data Analysis Plots in the HTML timeplot page.
    
        Args:
            None
            
        Returns:
            render_template(render_template): Render template for the plots  
        
    """
    df = pd.read_json("data/mini_sparkify_event_data.json", lines=True)
    
    cols_to_explore = get_number_of_distinct_records(df)

    graphs = [plot_category_counts(df, column_name) for column_name in cols_to_explore]
    graphs.append(new_users_plot(df))
    
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template("expanalysis.html", ids=ids, graphJSON=graphJSON)

@app.route("/featureengineering")
def featureengineering():
    """Generate Feature Engineering Plots in the HTML timeplot page.
    
        Args:
            None
            
        Returns:
            render_template(render_template): Render template for the plots  
        
    """
    days_churn_true = pd.read_csv("data/days_churn_true.csv")
    days_churn_true.drop("Unnamed: 0", axis=1, inplace=True)
    days_churn_false = pd.read_csv("data/days_churn_false.csv")
    days_churn_false.drop("Unnamed: 0", axis=1, inplace=True)

    churned_gender_counts = pd.read_csv("data/churned_gender_counts.csv")
    churned_gender_counts.drop("Unnamed: 0", axis=1, inplace=True)
    non_churned_gender_counts = pd.read_csv("data/non_churned_gender_counts.csv")
    non_churned_gender_counts.drop("Unnamed: 0", axis=1, inplace=True)

    churned_level_counts = pd.read_csv("data/churned_level_counts.csv")
    churned_level_counts.drop("Unnamed: 0", axis=1, inplace=True)
    non_churned_level_counts = pd.read_csv("data/non_churned_level_counts.csv")
    non_churned_level_counts.drop("Unnamed: 0", axis=1, inplace=True)

    graphs = [plot_churned_vs_non_churned_by_feature(churned_gender_counts, non_churned_gender_counts, 'gender'),
              plot_box_plot(days_churn_true, days_churn_false),
              plot_churned_vs_non_churned_by_feature(churned_level_counts, non_churned_level_counts, 'Level')
              ]
    
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template("featureengineering.html", ids=ids, graphJSON=graphJSON)

@app.route("/mlmodel")
def mlmodel():
    """Generate ML Model Results in the HTML timeplot page.
    
        Args:
            None
            
        Returns:
            render_template(render_template): Render template for the plots  
        
    """

    # Logistic Reression (LR) Evaluations
    coefs_df = pd.read_csv("data/coefs_df.csv")

    lr_roc_metrics_train = pd.read_csv("data/lr_training_roc.csv")
    lr_pr_metrics_train = pd.read_csv("data/lr_training_pr.csv")

    lr_roc_metrics_test = pd.read_csv("data/lr_test_roc_metrics.csv")
    lr_pr_metrics_test = pd.read_csv("data/lr_test_pr_metrics.csv")

    # Random Forest (RF) Evaluations
    rf_roc_metrics_train = pd.read_csv("data/rf_train_roc_metrics.csv")
    rf_pr_metrics_train = pd.read_csv("data/rf_train_pr_metrics.csv")

    rf_roc_metrics_test = pd.read_csv("data/rf_test_roc_metrics.csv")
    rf_pr_metrics_test = pd.read_csv("data/rf_test_pr_metrics.csv")

    
    graphs = [plot_coefs(coefs_df), 
              plot_roc_curve(lr_roc_metrics_train['FPR'], lr_roc_metrics_train['TPR'], 'Train Data - LR'),
              plot_pr_curve(lr_pr_metrics_train['recall'], lr_pr_metrics_train['precision'], 'Train Data - LR'),
              plot_roc_curve(lr_roc_metrics_test['FPR'], lr_roc_metrics_test['TPR'], 'Test Data - LR'),
              plot_pr_curve(lr_pr_metrics_test['Recall'], lr_pr_metrics_test['Precision'], 'Test Data - LR'),
              plot_roc_curve(rf_roc_metrics_train['FPR'], rf_roc_metrics_train['TPR'], 'Train Data - RF'),
              plot_pr_curve(rf_pr_metrics_train['Recall'], rf_pr_metrics_train['Precision'], 'Train Data - RF'),
              plot_roc_curve(rf_roc_metrics_test['FPR'], rf_roc_metrics_test['TPR'], 'Test Data - RF'),
              plot_pr_curve(rf_pr_metrics_test['Recall'], lr_pr_metrics_test['Precision'], 'Test Data - RF')
              ]
    
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template("mlmodel.html", ids=ids, graphJSON=graphJSON)

def plot_coefs(coefs_df):
    fig = go.Figure()

    # Positive bars with green color
    fig.add_trace(go.Bar(
        x=coefs_df['Feature_Name'][coefs_df['Feature_Coefficient'] >= 0],
        y=coefs_df['Feature_Coefficient'][coefs_df['Feature_Coefficient'] >= 0],
        marker_color='green',
        name='Positive Coefficients'
    ))

    # Negative bars with red color
    fig.add_trace(go.Bar(
        x=coefs_df['Feature_Name'][coefs_df['Feature_Coefficient'] < 0],
        y=coefs_df['Feature_Coefficient'][coefs_df['Feature_Coefficient'] < 0],
        marker_color='red',
        name='Negative Coefficients'
    ))

    fig.update_layout(
        title='Feature Coefficients (Diverging)',
        xaxis_title='Feature Name',
        yaxis_title='Feature Coefficient'
    )

    return fig

def plot_roc_curve(fpr, tpr, name):
     """Create a roc curve.

    Args:
        fpr: np array or list
        tpr: np array or list
        name: string
    Returns:
        None
    """
     fig_roc = go.Figure(data=go.Scatter(x=fpr, y=tpr, mode='lines', line=dict(color='darkorange', width=2), name='ROC Curve'))
    
     fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(color='navy', width=2, dash='dash'), name='Baseline'))
     fig_roc.update_layout(
        title=f'ROC Curve - ' + name,
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
    )
     return fig_roc

def plot_pr_curve(recall, precision, name):
     """Create a roc curve.

    Args:
        recall: np array or list
        precision: np array or list
        name: string
    Returns:
        None
    """
     fig_pr = go.Figure(data=go.Scatter(x=recall, y=precision, mode='lines', line=dict(color='darkorange', width=2), name='Precision-Recall Curve'))
     fig_pr.add_trace(go.Scatter(x=[0, 1], y=[1, 0], mode='lines', line=dict(color='navy', width=2, dash='dash'), name='Baseline'))

     fig_pr.update_layout(
            title=f'Precision-Recall Curve - '+name,
            xaxis_title='Recall',
            yaxis_title='Precision',
        )

    
     return fig_pr

def plot(df, missing_type='Null'):
    """Create a bar chart of the number of missing records of either 'Null' or 'Empty-Value' value
    in each column.

    Args:
        df (DataFrame): PySpark DataFrame object
        missing_type (string): can be either 'Null' or 'Empty-Value'
    Returns:
        None
    """

    df = df.iloc[0].loc[df.iloc[0] > 0].sort_values(ascending=False)

    x = [name.replace('_', ' ').capitalize() for name in df.index]
    y = df.values

    fig = go.Figure(data=[go.Bar(x=x, y=y, text=y, marker_color='rgb(55, 83, 109)', textposition='auto',
                                 hovertext=["Missing Records in the `{}` column: {}".format(bar, count) for bar, count in zip(x, y)],
                                 hoverinfo="text")])

    fig.update_layout(
        title=go.layout.Title(text="Number of {} Records per Column".format(missing_type), x=0.5),
        xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text="Column Name")),
        yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text="Counts"))
    )

    return fig

def plot_churned_vs_non_churned_by_feature(churned_df, non_churned_df, feature):
    """Create a bar chart comparing the counts of churned and non-churned users by feature.

    Args:
        churned_df (DataFrame): pandas DataFrame containing churned users with feature column.
        non_churned_df (DataFrame): pandas DataFrame containing non-churned users with feature column.

    Returns:
        None
    """

    trace_df = go.Bar(x=churned_df[feature], y=churned_df['count'], text=churned_df['count'], textposition='auto', name='Churned')
    trace_non_df = go.Bar(x=non_churned_df[feature], y=non_churned_df['count'], text=non_churned_df['count'], textposition='auto', name='Non_Churned')

    fig = go.Figure(data=[trace_df, trace_non_df])

    fig.update_layout(
        title='Counts of ' + feature,
        xaxis_title=feature,
        yaxis_title='Count',
        barmode='group'
    )


    return fig

def plot_box_plot(days_churn_true, days_churn_false):
    
    plotly_df = pd.concat([days_churn_true.assign(Churn='Churn True'), days_churn_false.assign(Churn='Churn False')])

    fig = px.box(plotly_df, x='Churn', y='Days', title='Days Values for Churn True and Churn False', points="all")
    
    return fig


def get_number_of_distinct_records(df, max_distinct_records=60):
    """Return a list of columns with at most max_distinct_records records.

    Args:
        df (DataFrame): pandas DataFrame object
        columns (list): a list of column names to analyze
        max_distinct_records (int): an upper bound on the number of distinct records

    Returns:
        cols (list): a list of column names with at most max_distinct_records distinct records
    """

    cols = []
    for column in df.columns:
        num_distinct_record = df[column].nunique()

        if num_distinct_record <= max_distinct_records:
            cols.append(column)

    return cols

def plot_category_counts(df, column_name):
    """Create a bar chart of the counts of different categories in a specific column.

    Args:
        df (DataFrame): pandas DataFrame object
        column_name (str): Name of the column to plot its category counts.

    Returns:
        None
    """
    # Calculate the counts of each category in the specified column
    category_counts = df[column_name].value_counts().reset_index()
    category_counts.columns = [column_name, "count"]
    category_counts.sort_values(by="count", ascending=False, inplace=True)

    # Prepare data for plotting
    x = category_counts[column_name].values
    y = category_counts["count"].values

    # Create a bar plot using Plotly
    fig = go.Figure(data=[go.Bar(x=x, y=y, text=y, marker_color='rgb(55, 83, 109)', textposition='auto',
                                 hovertext=["Category: {}<br>Count: {}".format(cat, count) for cat, count in zip(x, y)],
                                 hoverinfo="text")])

    # Customize the plot layout
    fig.update_layout(
        title=go.layout.Title(text="Counts of Different Categories in Column '{}'".format(column_name), x=0.5),
        xaxis=go.layout.XAxis(title=go.layout.xaxis.Title(text=column_name)),
        yaxis=go.layout.YAxis(title=go.layout.yaxis.Title(text="Counts"))
    )

    return fig

def new_users_plot(df):
    """Create a monthly bar chart for the number of new users in each month.
    
    Args:
        df (DataFrame): pandas DataFrame object containing 'registration' and 'userId' columns.
        
    Returns:
        None
    """
        
    # Drop rows with missing 'registration' values
    registrations = df[['registration', 'userId']].dropna()

    # Convert the 'registration' column to datetime format
    registrations['date'] = pd.to_datetime(registrations['registration'], unit='ms')

    # Resample and count new users per month
    count_series = registrations.resample('M', on='date').count()['registration']
    
    # Create a bar plot using Plotly
    fig = go.Figure(data=go.Bar(x=count_series.index, y=count_series.values))

    # Customize the plot layout
    fig.update_layout(
        xaxis_title='Months',
        yaxis_title='New Users',
        title='Number of New Users per Month'
    )

    return fig

def main():
    app.run(host="0.0.0.0", port=3001, debug=True)


if __name__ == "__main__":
    main()
