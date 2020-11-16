
# ds-practitioner-pod-cap
About AAVAIL Company:     

AAVAIL provides a streaming service similar to Netflix, Amazon Prime, they offer local, national and international news to its subscriber in 12 language. he news feed is piped through the service and both the speaker's voice and the movement of their lips are modified to match the language the subscriber has selected. There are separate deep-learning models for the image and audio portions of the service, but the experience is seamless from the standpoint of the user     Company engaged their sales and marketing team to increase campaign, modifying the pricing model, refining the product and more, with the goal of driving the product’s growth in the new markets Business scenario and testable hypotheses:     

As a data scientist, business scenario is to project a particular country's revenue for the following month using a machine learning model that will provide revenue estimates and confidence measures for those estimates.     

Company have sufficient customer invoice related data in well-structured format, which would need to analysis and process to build the forecasting model to achieve the required goal.


Project file details.

cs-train: Contains all the data to train the model
models: Contains all pre-trained saved models for prediction
notebooks: Contains all the notebooks describing solutions and depicting visualizations
templates: Simple templates for rendering flask app
unittest: It has logger test, API test and model test for testing all the functionalities before deploying to production and for maintenance post deployment
Dockerfile: Contains all the commands a user could call on the command line to assemble the docker image.
app.py: Flask app for creating a user interface /train and /predict APIs in order to train and predict respectively
cslib.py: A collection of functions that will transform the data set into features you can use to train a model.
model.py: A module having functions for training, loading a model and making predictions


Command for docker image.

    ~$ cd AI-workflow
    ~$ docker build -t capstone-project .

Check that the image is there.

    ~$ docker image ls

Run the container

docker run -p 4000:8080 ibmcapsproj-00832g

Project Review Points.

Unit tests for the API: unittests/ApiTests.py
Unit tests for the model: unittests/ModelTests.py
Unit tests for the logging: unittests/LoggerTests.py
Run all of the unit tests with a single script: run-tests.py
Read/write unit tests are isolated from production models and logs
APIs for training and prediction: /app.py
Data ingestion automation pipeline: /cslib.py
Multiple models comparison: notebooks/
EDA investigation with visualizations: notebooks/data_ingestion_eda_part1.ipynb
Containerization within a working Docker image: /Dockerfile
Visualization to compare the model to the baseline model: notebooks/time_series_iteration.ipynb
