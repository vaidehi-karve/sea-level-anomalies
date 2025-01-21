# sea-level-anomalies
UCSD SMASH &amp; NSF HDR ML Challenge Hackathon

Our project focused on predicting anomalies in Sea Level Anomalies (SLA) for 12 East Coast cities over the next 10 years. To do this, we turned .nc files into a design matrix that could be used for machine learning. Each .nc file provided data for one day, including city, latitude, longitude, and SLA metrics (average, maximum, and minimum). These metrics were calculated using SLA values from nearby points within a 3-degree area.

We faced challenges working with .nc files since we had no prior experience with them. Extracting SLA data was tricky because it was stored as a MaskedArray, which we had to learn how to handle. Despite this, we eventually succeeded in creating a dataset with all the required features.

Initially, we used a KNN classifier but found it unsuitable, as KNN struggled to detect anomalies because the dataset had many similar points, and most data showed no anomalies. It defaulted to predicting “no anomaly” most of the time. We then switched to a Random Forest Classifier (RFC), which worked better. RFC uses decision trees with thresholds to split data, making it effective for our numerical features (average, maximum, and minimum SLA).

We split the data into 75% training and 25% testing. The RFC achieved 99% accuracy on the training set and 97.7% accuracy on the test set, showing it worked well without overfitting. Our model accurately predicts SLA anomalies and performs well with high accuracy. Despite initial challenges, learning how to process .nc files and choosing the right model were key to our success.
