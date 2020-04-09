# DisasterRecovery-NDUdacity
DisasterRecovery project for the Data Scientist ND of Udacity


## Project Overview

Starting fron the disaster data provided by <a href="https://www.figure-eight.com/" target="_blank">Figure Eight</a>, the aim of this project is to build an engine to be used to classify disaster messages.

The engine was tested with an web app that gives the chance to an emergengy worker to type an help request and receive the related classification.

## Project Components

There are three components of this project: ETL Pipeline,  ML Pipeline, Flask Web App

### ETL Pipeline

File _data/process_data.py_ contains data cleaning pipeline that loads the messages and categories dataset, merges the two datasets, cleans the data ad finally stores it in a SQLite database

### ML Pipeline

File _models/train_classifier.py_ contains machine learning pipeline that loads data from the SQLite database, splits the data into training and testing sets, builds a text processing and machine learning pipeline, trains and tunes a model using GridSearchCV, outputs result on the test set, exports the final model as a pickle file

### Flask Web App

If you run the proper command from app directory you will start the web app where users can enter their the message. The app then classifies the text message into categories sto be used to define which agency have to be caled for the proper support.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
        It takes time to execute,please be patient!

2. Run the following command in the app's directory to run your web app
    `python run.py`

3. Open another Terminal Window and type 
    `env|grep WORK`

4. Take note of the WORKSPACEID string and in a new web browser window, type in the following:
    `https://WORKSPACEID-3001.udacity-student-workspaces.com/`, changing `WORKSPACEID` with the proper value.
    
    At this point you should be able to see the web app. The number 3001 represents the port where your web app will show up. 
    Make sure that the 3001 is part of the web address you type in.

## Credits and Acknowledgements

Thanks to Udacity for giving me the chance to develope this project!!!
