from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns
from sklearn.feature_selection import RFE
from imblearn.over_sampling import SMOTE
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, 
LinearDiscriminantAnalysis

# Load the data
train_data = pd.read_csv("training_data.csv")
X = train_data.drop(columns=['label'])
y = train_data['label']
test_data = pd.read_csv("songs_to_classify.csv")

# Print data info
print(train_data.shape)
print(train_data.head())
print(train_data.info())
print(train_data.describe())

# Balance the dataset using SMOTE
counter = Counter(y)
print(counter)
oversample = SMOTE(random_state=0)
X, y = oversample.fit_resample(X, y)
counter = Counter(y)
print(counter)

# Preprocessing function
def get_preprocessed_data(data):
    categorical_features = ['key', 'mode']
    numeric_features = list(data.columns.drop(categorical_features))
    preprocessed_data = pd.get_dummies(data, prefix=categorical_features, 
columns=categorical_features)
    scaler = MinMaxScaler()
    scaler.fit(preprocessed_data[numeric_features])
    preprocessed_data[numeric_features] = 
scaler.transform(preprocessed_data[numeric_features])
    return preprocessed_data

# Preprocess the data
preprocessed_X = get_preprocessed_data(X)
preprocessed_test = get_preprocessed_data(test_data)

# Feature selection using RFE
rfe = RFE(estimator=RandomForestClassifier(max_depth=4, random_state=0), 
n_features_to_select=5)
rfe = rfe.fit(preprocessed_X, y)
print(rfe.support_)
print(rfe.ranking_)
print(X.columns[rfe.support_])

# Correlation Heatmap
corrmat = preprocessed_X.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20, 20))
sns.heatmap(preprocessed_X[top_corr_features].corr(), annot=True, cmap="RdYlGn")
plt.show()

# Function to get needed features
def get_needed_features(data_set):
    return data_set.loc[:, ['acousticness', 'speechiness', 'loudness', 'energy', 
'danceability']]

# Get needed features
preprocessed_X = get_needed_features(preprocessed_X)
preprocessed_test = get_needed_features(preprocessed_test)

# Function to draw ROC curve
def draw_ROC(y_test, prediction, pred_probabilities):
    logit_roc_auc = roc_auc_score(y_test, prediction)
    fpr, tpr, thresholds = roc_curve(y_test, pred_probabilities)
    plt.figure()
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1.1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    plt.show()

# Function to get string prediction
def get_string_prediction(prediction):
    return ''.join([str(x) for x in list(prediction)])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(preprocessed_X, y, test_size=0.2, 
random_state=2)

# Train Logistic Regression model
model = LogisticRegression().fit(X_train, y_train)
prediction_prob = model.predict_proba(X_test)[:, 1]
false_positives_rate, true_positive_rates, thresholds = roc_curve(y_test, 
prediction_prob)
threshold_true_positive = 0

# Get the threshold for true positive rate of 100%
for index, true_positive_rate in enumerate(true_positive_rates):
    if true_positive_rate == 1:
        threshold_true_positive = thresholds[index]
        break

always_like_prediction = (prediction_prob > threshold_true_positive).astype(int)

# Draw ROC curve
draw_ROC(y_test, always_like_prediction, prediction_prob)

print(f"The threshold used is {threshold_true_positive}")
print(f"Misclassification for always like prediction model: 
{np.mean(always_like_prediction != y_test)}")
print(pd.crosstab(y_test, always_like_prediction))
print(confusion_matrix(y_test, always_like_prediction))

# Define models
MODELS = {
    'LogisticRegression': {
        "has_hyperparameters": False,
        "misclassification": 1
    },
    'QuadraticDiscriminantAnalysis': {
        "has_hyperparameters": False,
        "misclassification": 1
    },
    'LinearDiscriminantAnalysis': {
        "has_hyperparameters": False,
        "misclassification": 1
    },
    'KNeighborsClassifier': {
        "has_hyperparameters": True,
        # K
        "hyperparameters": np.arange(1, 100),
        "best_hyperparameter": 1,
        "misclassification": 1
    },
    'RandomForestClassifier': {
        "has_hyperparameters": True,
        # depth
        "hyperparameters": np.arange(1, 30),
        "best_hyperparameter": 1,
        "misclassification": 1
    }
}

# Cross-validation to get best hyperparameters
n_fold = 5
cv = KFold(n_splits=n_fold, random_state=2, shuffle=True)

# Go through each model
for model_name, model_info in MODELS.items():
    if model_info['has_hyperparameters']:
        hyperparameter_values = model_info['hyperparameters']
        misclassification = np.zeros(len(hyperparameter_values))
    else:
        misclassification = 0
    
    for train_index, val_index in cv.split(X_train):
        X_train_, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_, y_val = y_train.iloc[train_index], y_train.iloc[val_index]
        
        if model_name == 'LogisticRegression':
            model = LogisticRegression()
        elif model_name == 'QuadraticDiscriminantAnalysis':
            model = QuadraticDiscriminantAnalysis()
        elif model_name == 'LinearDiscriminantAnalysis':
            model = LinearDiscriminantAnalysis()
        
        if not model_info['has_hyperparameters']:
            model.fit(X_train_, y_train_)
            prediction = model.predict(X_val)
            misclassification += np.mean(prediction != y_val)
            continue
        
        for index, value in enumerate(hyperparameter_values):
            if model_name == 'KNeighborsClassifier':
                model = KNeighborsClassifier(n_neighbors=value)
            elif model_name == 'RandomForestClassifier':
                model = RandomForestClassifier(max_depth=value, max_features='sqrt', 
bootstrap=True, random_state=0)
            
            model.fit(X_train_, y_train_)
            prediction = model.predict(X_val)
            misclassification[index] += np.mean(prediction != y_val)
    
    misclassification /= n_fold
    
    if not model_info['has_hyperparameters']:
        least_misclassification = misclassification
    else:
        least_misclassification = min(misclassification)
        best_hyperparameter = 
hyperparameter_values.tolist()[misclassification.tolist().index(least_misclassification)]
    
    MODELS[model_name]['misclassification'] = least_misclassification
    
    if model_info['has_hyperparameters']:
        MODELS[model_name]['best_hyperparameter'] = best_hyperparameter
    
    if not model_info['has_hyperparameters']:
        print(f"Model {model_name}: \n Misclassification: {least_misclassification}")
    else:
        print(f"Model {model_name}:\n Hyperparameter: {best_hyperparameter}\n 
Misclassification: {least_misclassification}")

# Use the model with the lowest misclassification
best_model_name = ''
least_misclassification = 1

for model_name, model_info in MODELS.items():
    if least_misclassification > model_info['misclassification']:
        best_model_name = model_name
        least_misclassification = model_info['misclassification']

if best_model_name == 'LogisticRegression':
    model = LogisticRegression()
elif best_model_name == 'QuadraticDiscriminantAnalysis':
    model = QuadraticDiscriminantAnalysis()
elif best_model_name == 'LinearDiscriminantAnalysis':
    model = LinearDiscriminantAnalysis()
elif best_model_name == 'KNeighborsClassifier':
    model = 
KNeighborsClassifier(n_neighbors=MODELS[best_model_name]['best_hyperparameter'])
elif best_model_name == 'RandomForestClassifier':
    model = 
RandomForestClassifier(max_depth=MODELS[best_model_name]['best_hyperparameter'], 
bootstrap=True, max_features='sqrt', random_state=0)

model.fit(X_train, y_train)

prediction = model.predict(X_test)
print(f"Classifier used is {best_model_name}")
print(f"Misclassification on test data: {np.mean(prediction != y_test)}")
print(pd.crosstab(y_test, prediction))

# Use all the data to train the model and make the prediction
model.fit(preprocessed_X, y)
prediction = model.predict(preprocessed_test)
print(get_string_prediction

