import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import netCDF4
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

class Model:
    def __init__(self):
        print("Our Model initialized")
        # Put your initialization code here
        # load the save model here
        
    def predict(self, X):
        # Put your prediction code here
        # This example predicts a random value for 12 station
        # The output should be a dataframe with 10 rows and 12 columns
        # Each value should be 1 for anamoly and 0 for normal
        # Return a np array of 1s and 0s with the same length of 12
        # with random prediction of 1 or 0

        # Preprocessing 

        # Background information
        directory = r"/Users/vaidehikarve/Downloads/Sea Level Anomalies Hackathon/Copernicus_ENA_Satelite Maps_Training_Data"
        cities = [
            "Atlantic City",
            "Baltimore",
            "Eastport",
            "Fort Pulaski",
            "Lewes",
            "New London",
            "Newport",
            "Portland",
            "Sandy Hook",
            "Sewells Point",
            "The Battery",
            "Washington"]
        
        city_locations = {
            'Fort Pulaski': (32.0367, -80.9017),
            'Sewells Point': (36.946701, -76.330002),
            'Lewes': (38.78278, -75.119164),
            'Washington': (38.873, -77.0217),
            'Baltimore': (39.266944, -76.579444),
            'Atlantic City': (39.356667, -74.418053),
            'Sandy Hook': (40.466944, -74.009444),
            'The Battery': (40.700556, -74.014167),
            'New London': (41.361401, -72.089996),
            'Newport': (41.504333, -71.326139),
            'Portland': (43.65806, -70.24417),
            'Eastport': (44.904598, -66.982903)
        }
        
        all_anomalies = pd.read_csv("/Users/vaidehikarve/Downloads/Sea Level Anomalies Hackathon/Training_Anomalies_Station Data/training_anomalies_merged.csv")
        
        # Preprocessing the training dataset
        # Internal functions 
        def create_features(sla_df, city, date, a):
            area_df = area_sla(sla_df, city, a)
            avg = np.mean(area_df)
            max = np.max(area_df)
            min = np.min(area_df)
            lat = city_locations[city][0]
            long = city_locations[city][1]
            if all_anomalies[(all_anomalies['t'] == date) & (all_anomalies['location'] == city)].shape[0] == 1:
                anomaly = 1
            else:
                anomaly = 0
            return [city, avg, max, min, lat, long, date, anomaly]
        
        def area_sla(sla_df, city, a):
            lat = city_locations[city][0]
            long = city_locations[city][1]
            adj = sla_df[sla_df.index > lat - a]
            adj = adj[adj.index< lat + a]
            return adj[list(filter(lambda lat: lat < long + a and lat > long - a, adj.columns))].replace(-2147483647, np.nan)
        
        results = []
        for filename in os.listdir(directory):
            if filename.endswith(".nc"):
                file_path = os.path.join(directory, filename)

                # Extract the date part from the filename and format it
                date_str = filename.split(".")[0]
                if len(date_str) == 23:
                    formatted_date = f"{date_str[7:11]}-{date_str[11:13]}-{date_str[13:15]}"

                    # Open the .nc file
                    dataset = netCDF4.Dataset(file_path, mode="r")

                    latitude = dataset.variables["latitude"][:]
                    longitude = dataset.variables["longitude"][:]

                    # Extract the 'sla' variable
                    sla = dataset.variables["sla"][:]
                    sla_df = pd.DataFrame(sla.data[0], index = latitude, columns= longitude)

                    # Create DF
                    for city in cities:
                        results.append(create_features(sla_df, city, formatted_date, 3))

                    # Close the dataset
                    dataset.close()
        
        # Training set
        X_y_train = pd.DataFrame(results)
        X_y_train = X_y_train.rename(columns={0: 'City', 1: 'Avg SLA', 2:'Max SLA', 3:'Min SLA', 4:'Latitude', 5: 'Longitude', 6:'Date', 7: 'Anomaly'})
        
        X_y_train['Date'] = pd.to_datetime(X_y_train['Date'])


        # Preprocessing the evaluation set

        def create_features_eval(sla_df, city, date, a): # Without anomaly
            area_df = area_sla(sla_df, city, a)
            avg = np.mean(area_df)
            max = np.max(area_df)
            min = np.min(area_df)
            lat = city_locations[city][0]
            long = city_locations[city][1]
            return [city, avg, max, min, lat, long, date]

        eval_results = []
        for filename in os.listdir(X):
            if filename.endswith(".nc"):
                file_path = os.path.join(X, filename)

                # Extract the date part from the filename and format it
                date_str = filename.split(".")[0]
                if len(date_str) == 23:
                    formatted_date = f"{date_str[7:11]}-{date_str[11:13]}-{date_str[13:15]}"

                    # Open the .nc file
                    eval_dataset = netCDF4.Dataset(file_path, mode="r")

                    latitude = eval_dataset.variables["latitude"][:]
                    longitude = eval_dataset.variables["longitude"][:]

                    # Extract the 'sla' variable
                    sla = eval_dataset.variables["sla"][:]
                    sla_df = pd.DataFrame(sla.data[0], index = latitude, columns= longitude)

                    # Create DF
                    for city in cities:
                        eval_results.append(create_features_eval(sla_df, city, formatted_date, 3))

                    # Close the dataset
                    eval_dataset.close()

        X_eval = pd.DataFrame(eval_results)
        X_eval = X_eval.rename(columns={0: 'City', 1: 'Avg SLA', 2:'Max SLA', 3:'Min SLA', 4:'Latitude', 5: 'Longitude', 6:'Date'})
        
        X_eval['Date'] = pd.to_datetime(X_eval['Date']) # Evaluation DF

        # Model fit and prediction
        def date_trans(df):
            return pd.DataFrame({
                "Month": df["Date"].apply(lambda x: x.month),
                "Year": df["Date"].apply(lambda x: x.year)
            })

        preproc = ColumnTransformer(
            transformers = [
                ("categorical", OneHotEncoder(drop='first', handle_unknown='ignore'), ["City"]),
                ("date", FunctionTransformer(date_trans), ["Date"])
            ],
            remainder='passthrough'
        )

        X_train = X_y_train.drop("Anomaly", axis=1)
        y_train = X_y_train["Anomaly"]

        pd.DataFrame(preproc.fit_transform(X_train))

        rfc = RandomForestClassifier(criterion = 'entropy', max_features = None)


        model = Pipeline([("preproc", preproc), ("classifier", rfc)])
        model.fit(X_train, y_train)

        preds = model.predict(X_eval)
        return preds