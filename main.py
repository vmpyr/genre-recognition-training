import pandas as pd
from sklearn.model_selection import train_test_split

# reading the csv file
df = pd.read_csv('music_train.csv')

# interchanging the topic and genre columns
feature_names = list(df.columns)
feature_names[-1], feature_names[-2] = feature_names[-2], feature_names[-1]
df = df[feature_names]

# filling the empty NaN spaces with the median of the respective column
df = df.fillna(df.median())

# interpreting out the feature values and target values
feature = df.iloc[:, 1:-2].values
targets = df.iloc[:, -1].values

# utlizing 20% of the total data for testing
# x_train, x_test, y_train, y_test = train_test_split(feature, targets, test_size=0.20, random_state=0)

# training the model for finding the target value by the Random Forest Classification algorithm
from sklearn.ensemble import RandomForestClassifier
md = RandomForestClassifier(n_estimators=10000)
md.fit(feature, targets)

# importing the given test csv file
test_file = pd.read_csv('music_test.csv')

# filling out the NaN spaces in the given test file
test_file = test_file.fillna(test_file.median())

# interpreting out features of the given test file which we have trained the machine for
test_feature = test_file.iloc[:, 1:-1]

# finding and printing out the predicted values
y_pred_test = md.predict(test_feature)
print(y_pred_test)

# putting together the 'id' and 'predictions'
submission = {'id': test_file['id'], 'genre': y_pred_test}

# changing prediction data from numpy array to panda dataframe
prediction = pd.DataFrame(submission)

# changing the prediction to strictly integer type
prediction = prediction.astype(int)

# exporting the final predicted csv file
prediction.to_csv('submission.csv', index=False)
