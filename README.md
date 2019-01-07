# Disaster Response Pipeline Project

## Summary of this project

This is my 4th project of Udacity Data Scientist Nanodegree.

This project's goal is to make a model and platform to judge what type of information is contained in the short messages from people in the middle of a disaster. The model was trained by dataset of [Figure Eight - Multilingual Disaster Response Messages](https://www.figure-eight.com/dataset/combined-disaster-response-data/).

Imagine the situation where an earthquake occurs. Many people tweets or send messages from many kinds of platforms. These sentences implies the situation they are in, and also the aid they need (i.e. wrecked buildings / necessity of water / help for unconscious person in front of them..) These messages should be dealt with instantaneously (ideally automatically) to be sent to appropriate disaster relief agencies.

To achieve this purpose, I constructed a model to classify which type of information the brief message contains. This information consists of 35 labels which are listed in the *Dataset Detail* below. Even though the  original dataset contains 36 types of labels, there is no message which is tagged as 'child_alone'. So it was impossible to make the model to judge whether a message given is about 'child_alone'.  

I built a demonstration web site of this model. Following the instruction above, you can see a simple web site. This web site has 2 functions,

1. Classification of the message entered by a user.  
When you enter a sentence you come up with in a search box, the labels which the message is predicted to contain are shown. This helps you understand how precisely the model in this project classifies messages.

2. Visualization of distribution of labels of the training dataset.  
This portion shows the distribution of 3 labels "related", "request", "offer". This gives you a little idea about the dataset which this project classifier was built on.




## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/



## Repo Structure

    .
    ├── README.md
    ├── app
    │   ├── run.py
    │   └── templates
    │       ├── go.html
    │       └── master.html
    ├── data
    │   ├── DisasterResponse.db       # database to save clean data to
    │   ├── disaster_categories.csv   # data to process
    │   ├── disaster_messages.csv     # data to process
<<<<<<< HEAD
    │   └── cleandata.py
    ├── draft_ipynb                   # notebooks for preparation)
    │   ├── ETLPipelinePreparation.ipynb
=======
    │   └── process_data.py
    ├── draft_ipynb                   # notebooks for preparation (ignore!)
    │   ├── ETLPipelinePreparation.ipynb
>>>>>>> 02d0059174de0770a7f7334e21d2ccf0402c12fd
    │   ├── ExploringMemo.ipynb
    │   ├── MLPipelinePreparation.ipynb
    │   ├── disasters.db
    |   └── clf.pkl
    └── models
        ├── classifier.pkl            # saved model
        └── train_classifier.py



#### data/disaster_messages.csv

A csv file which contains real messages that were sent from people facing disaster events, and the other which tells whether the messages is about the topic of each 36 types of information.

        'related', 'request', 'offer', 'aid_related', 'medical_help',
        'medical_products', 'search_and_rescue', 'security', 'military',
        'child_alone', 'water', 'food', 'shelter', 'clothing', 'money',
        'missing_people', 'refugees', 'death', 'other_aid',
        'infrastructure_related', 'transport', 'buildings', 'electricity',
        'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
        'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
        'other_weather', 'direct_report']

#### data/disaster_categories.csv

A csv file which tells which types of information the messages contained (That does not mean that 1 message contains only 1 label of information. For example, it can be case that 1 messages contains 2 types of information, which are 'genres', 'offer', 'aid_related')
