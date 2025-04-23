def candidate_elimination_algorithm(training_data):
    # Initialize the sets of hypotheses
    S = [['?' for _ in range(len(training_data[0]) - 1)]]
    G = [['?' for _ in range(len(training_data[0]) - 1)]]

    # Iterate through the training examples
    for example in training_data:
        # Check if the example is positive
        if example[-1] == 'Yes':
            # Remove from S any hypothesis that is not consistent with the example
            S = [hypothesis for hypothesis in S if is_consistent(hypothesis, example)]

            # Generalize hypotheses in S by removing any attribute that is different from the example
            for i in range(len(S)):
                for j in range(len(S[i])):
                    if S[i][j] != example[j]:
                        S[i][j] = '?'

            # Remove from G any hypothesis that is more general than a hypothesis in S
            G = [hypothesis for hypothesis in G if not any_more_general(hypothesis, S)]

        else:
            # Remove from G any hypothesis that is consistent with the example
            G = [hypothesis for hypothesis in G if not is_consistent(hypothesis, example)]

            # Specialize hypotheses in G by replacing any attribute that is the same as the example with a wildcard
            for i in range(len(G)):
                for j in range(len(G[i])):
                    if G[i][j] == example[j]:
                        G[i][j] = '?'

            # Remove from S any hypothesis that is more specific than a hypothesis in G
            S = [hypothesis for hypothesis in S if not any_more_specific(hypothesis, G)]

    return S, G


def is_consistent(hypothesis, example):
    for i in range(len(hypothesis)):
        if hypothesis[i] != '?' and hypothesis[i] != example[i]:
            return False
    return True


def any_more_general(hypothesis, hypotheses):
    for h in hypotheses:
        if all(x == '?' or x == y for x, y in zip(hypothesis, h)):
            return True
    return False


def any_more_specific(hypothesis, hypotheses):
    for h in hypotheses:
        if all(x == '?' or x == y for x, y in zip(h, hypothesis)):
            return True
    return False


# Sample training data
training_data = [
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
]

# Apply the Candidate-Elimination algorithm
S, G = candidate_elimination_algorithm(training_data)

# Print the set of most specific hypotheses
print("Most specific hypotheses:")
for hypothesis in S:
    print(hypothesis)

# Print the set of most general hypotheses
print("Most general hypotheses:")
for hypothesis in G:
    print(hypothesis)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import pandas as pd
from google.colab import files
uploaded = files.upload()



import io
df= pd.read_csv(io.BytesIO(uploaded['salaries.csv']))
df
df.head()

inputs = df.drop('salary_more_then_100k',axis='columns')


target = df['salary_more_then_100k']

from sklearn.preprocessing import LabelEncoder
le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()


inputs['company_n'] = le_company.fit_transform(inputs['company'])
inputs['job_n'] = le_job.fit_transform(inputs['job'])
inputs['degree_n'] = le_degree.fit_transform(inputs['degree'])



inputs_n = inputs.drop(['company','job','degree'],axis='columns')

target


from sklearn import tree
model = tree.DecisionTreeClassifier()

model.fit(inputs_n, target)

model.score(inputs_n,target)

model.predict([[2,1,0]])

model.predict([[2,1,1]])

model.fit(inputs_n, target)


model.score(inputs_n,target)


model.predict([[2,1,0]])


import pandas as pd

# Define the input data with feature names
input_data = pd.DataFrame([[2, 1, 0]], columns=['company_n', 'job_n', 'degree_n'])

# Make predictions
prediction = model.predict(input_data)
print("Prediction:", prediction)


import pandas as pd

# Define the input data with feature names
input_data = pd.DataFrame([[2, 1, 1]], columns=['company_n', 'job_n', 'degree_n'])

# Make predictions
prediction = model.predict(input_data)
print("Prediction:", prediction)


import pandas as pd

# Define the input data with feature names
input_data = pd.DataFrame([[2, 2, 1]], columns=['company_n', 'job_n', 'degree_n'])

# Make predictions
prediction = model.predict(input_data)
print("Prediction:", prediction)


+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import pandas as pd
from sklearn.datasets import load_digits
digits = load_digits()

dir(digits)


%matplotlib inline
import matplotlib.pyplot as plt

plt.gray()
for i in range(4):
    plt.matshow(digits.images[i])

df = pd.DataFrame(digits.data)
df.head()

df['target'] = digits.target

df[0:12]


X = df.drop('target',axis='columns')
y = df.target


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=20)
model.fit(X_train, y_train)


model.score(X_test, y_test)

y_predicted = model.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)
cm


%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load dataset
data = load_iris()

dir(data)

# Get features and target
X=data.data
y=data.target

y

# Get dummy variable
y = pd.get_dummies(y).values


y[:3]

#Split data into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=20, random_state=4)
.

# Initialize variables
learning_rate = 0.1
iterations = 5000
N = y_train.size

# number of input features
input_size = 4

# number of hidden layers neurons
hidden_size = 2

# number of neurons at the output layer
output_size = 3

results = pd.DataFrame(columns=["mse", "accuracy"])

# Initialize weights
np.random.seed(10)


# initializing weight for the hidden layer
W1 = np.random.normal(scale=0.5, size=(input_size, hidden_size))

# initializing weight for the output layer
W2 = np.random.normal(scale=0.5, size=(hidden_size , output_size))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def mean_squared_error(y_pred, y_true):
    return ((y_pred - y_true)**2).sum() / (2*y_pred.size)


def accuracy(y_pred, y_true):
    acc = y_pred.argmax(axis=1) == y_true.argmax(axis=1)
    return acc.mean()

for itr in range(iterations):

    # feedforward propagation
    # on hidden layer
    Z1 = np.dot(X_train, W1)
    A1 = sigmoid(Z1)

    # on output layer
    Z2 = np.dot(A1, W2)
    A2 = sigmoid(Z2)


    # Calculating error
    mse = mean_squared_error(A2, y_train)
    acc = accuracy(A2, y_train)
    results = pd.concat([results, pd.DataFrame([{"mse": mse, "accuracy": acc}])], ignore_index=True)


    # backpropagation
    E1 = A2 - y_train
    dW1 = E1 * A2 * (1 - A2)

    E2 = np.dot(dW1, W2.T)
    dW2 = E2 * A1 * (1 - A1)


    # weight updates
    W2_update = np.dot(A1.T, dW1) / N
    W1_update = np.dot(X_train.T, dW2) / N

    W2 = W2 - learning_rate * W2_update
    W1 = W1 - learning_rate * W1_update


results.mse.plot(title="Mean Squared Error")

results.accuracy.plot(title="Accuracy")

Z1 = np.dot(X_test, W1)
A1 = sigmoid(Z1)


Z2 = np.dot(A1, W2)
A2 = sigmoid(Z2)

acc = accuracy(A2, y_test)
print("Accuracy: {}".format(acc))
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import pandas as pd

from google.colab import files
uploaded = files.upload()

import io
df = pd.read_csv(io.BytesIO(uploaded['titanic.csv']))
# Dataset is now stored in a Pandas Dataframe
df.head()

df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis='columns',inplace=True)
df.head()

inputs = df.drop('Survived',axis='columns')
target = df.Survived

#inputs.Sex = inputs.Sex.map({'male': 1, 'female': 2})

dummies = pd.get_dummies(inputs.Sex)
dummies.head(3)

inputs = pd.concat([inputs,dummies],axis='columns')
inputs.head(3)

inputs.drop(['Sex','male'],axis='columns',inplace=True)
inputs.head(3)


inputs.columns[inputs.isna().any()]

inputs.Age[:10]

inputs.Age = inputs.Age.fillna(inputs.Age.mean())
inputs.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(inputs,target,test_size=0.3)

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()

model.fit(X_train,y_train)

model.score(X_test,y_test)

X_test[0:10]


y_test[0:10]


model.predict(X_test[0:10])

model.predict_proba(X_test[:10])

from sklearn.model_selection import cross_val_score
cross_val_score(GaussianNB(),X_train, y_train, cv=5)

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import pandas as pd

from google.colab import files
uploaded = files.upload()



!pip install chardet

import chardet
import pandas as pd
import io


encoding = chardet.detect(uploaded['spam1.csv'])['encoding']

df2 = pd.read_csv(io.BytesIO(uploaded['spam1.csv']), encoding=encoding)
df2

df2.groupby('Category').describe()

df2['spam']=df2['Category'].apply(lambda x: 1 if x=='spam' else 0)
df2.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df2.Message,df2.spam)

from sklearn.feature_extraction.text import CountVectorizer
v = CountVectorizer()
X_train_count = v.fit_transform(X_train.values)
X_train_count.toarray()[:2]

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train_count,y_train)

emails = [
    'As per your request Melle Melle (Oru Minnaminunginte Nurungu Vettam)has been set as your callertune for all Callers. Press *9 to copy your friends Callertune',
    'Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&Cs apply 08452810075over18s'
]

emails_count = v.transform(emails)
model.predict(emails_count)

X_test_count = v.transform(X_test)
model.score(X_test_count, y_test)

from sklearn.pipeline import Pipeline
clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])

clf.fit(X_train, y_train)

clf.score(X_test,y_test)

clf.predict(emails)



++++++++++++++++++++++++++++++++++++++++++++
import pandas as pd

from google.colab import files
uploaded = files.upload()



!pip install chardet

import chardet
import pandas as pd
import io


encoding = chardet.detect(uploaded['spam1.csv'])['encoding']

df2 = pd.read_csv(io.BytesIO(uploaded['spam1.csv']), encoding=encoding)
df2

df2.groupby('Category').describe()

df2['spam']=df2['Category'].apply(lambda x: 1 if x=='spam' else 0)
df2.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df2.Message,df2.spam)

from sklearn.feature_extraction.text import CountVectorizer
v = CountVectorizer()
X_train_count = v.fit_transform(X_train.values)
X_train_count.toarray()[:2]

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train_count,y_train)

emails = [
    'As per your request Melle Melle (Oru Minnaminunginte Nurungu Vettam)has been set as your callertune for all Callers. Press *9 to copy your friends Callertune',
    'Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&Cs apply 08452810075over18s'
]

emails_count = v.transform(emails)
model.predict(emails_count)

X_test_count = v.transform(X_test)
model.score(X_test_count, y_test)

from sklearn.pipeline import Pipeline
clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])

clf.fit(X_train, y_train)

clf.score(X_test,y_test)

clf.predict(emails)



+++++++++++++

ML exp 7.ipynb_

Aim:Write a program to construct a Bayesian network considering medical data. Use this model to demonstrate the diagnosis of heart patients using standard Heart Disease Data Set?


!pip install pgmpy

Collecting pgmpy
  Downloading pgmpy-0.1.26-py3-none-any.whl.metadata (9.1 kB)
Requirement already satisfied: networkx in /usr/local/lib/python3.11/dist-packages (from pgmpy) (3.4.2)
Requirement already satisfied: numpy in /usr/local/lib/python3.11/dist-packages (from pgmpy) (1.26.4)
Requirement already satisfied: scipy in /usr/local/lib/python3.11/dist-packages (from pgmpy) (1.13.1)
Requirement already satisfied: scikit-learn in /usr/local/lib/python3.11/dist-packages (from pgmpy) (1.6.1)
Requirement already satisfied: pandas in /usr/local/lib/python3.11/dist-packages (from pgmpy) (2.2.2)
Requirement already satisfied: pyparsing in /usr/local/lib/python3.11/dist-packages (from pgmpy) (3.2.1)
Requirement already satisfied: torch in /usr/local/lib/python3.11/dist-packages (from pgmpy) (2.5.1+cu121)
Requirement already satisfied: statsmodels in /usr/local/lib/python3.11/dist-packages (from pgmpy) (0.14.4)
Requirement already satisfied: tqdm in /usr/local/lib/python3.11/dist-packages (from pgmpy) (4.67.1)
Requirement already satisfied: joblib in /usr/local/lib/python3.11/dist-packages (from pgmpy) (1.4.2)
Requirement already satisfied: opt-einsum in /usr/local/lib/python3.11/dist-packages (from pgmpy) (3.4.0)
Requirement already satisfied: xgboost in /usr/local/lib/python3.11/dist-packages (from pgmpy) (2.1.3)
Requirement already satisfied: google-generativeai in /usr/local/lib/python3.11/dist-packages (from pgmpy) (0.8.4)
Requirement already satisfied: google-ai-generativelanguage==0.6.15 in /usr/local/lib/python3.11/dist-packages (from google-generativeai->pgmpy) (0.6.15)
Requirement already satisfied: google-api-core in /usr/local/lib/python3.11/dist-packages (from google-generativeai->pgmpy) (2.19.2)
Requirement already satisfied: google-api-python-client in /usr/local/lib/python3.11/dist-packages (from google-generativeai->pgmpy) (2.155.0)
Requirement already satisfied: google-auth>=2.15.0 in /usr/local/lib/python3.11/dist-packages (from google-generativeai->pgmpy) (2.27.0)
Requirement already satisfied: protobuf in /usr/local/lib/python3.11/dist-packages (from google-generativeai->pgmpy) (4.25.5)
Requirement already satisfied: pydantic in /usr/local/lib/python3.11/dist-packages (from google-generativeai->pgmpy) (2.10.5)
Requirement already satisfied: typing-extensions in /usr/local/lib/python3.11/dist-packages (from google-generativeai->pgmpy) (4.12.2)
Requirement already satisfied: proto-plus<2.0.0dev,>=1.22.3 in /usr/local/lib/python3.11/dist-packages (from google-ai-generativelanguage==0.6.15->google-generativeai->pgmpy) (1.25.0)
Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.11/dist-packages (from pandas->pgmpy) (2.8.2)
Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.11/dist-packages (from pandas->pgmpy) (2024.2)
Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.11/dist-packages (from pandas->pgmpy) (2025.1)
Requirement already satisfied: threadpoolctl>=3.1.0 in /usr/local/lib/python3.11/dist-packages (from scikit-learn->pgmpy) (3.5.0)
Requirement already satisfied: patsy>=0.5.6 in /usr/local/lib/python3.11/dist-packages (from statsmodels->pgmpy) (1.0.1)
Requirement already satisfied: packaging>=21.3 in /usr/local/lib/python3.11/dist-packages (from statsmodels->pgmpy) (24.2)
Requirement already satisfied: filelock in /usr/local/lib/python3.11/dist-packages (from torch->pgmpy) (3.17.0)
Requirement already satisfied: jinja2 in /usr/local/lib/python3.11/dist-packages (from torch->pgmpy) (3.1.5)
Requirement already satisfied: fsspec in /usr/local/lib/python3.11/dist-packages (from torch->pgmpy) (2024.10.0)
Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.1.105 in /usr/local/lib/python3.11/dist-packages (from torch->pgmpy) (12.1.105)
Requirement already satisfied: nvidia-cuda-runtime-cu12==12.1.105 in /usr/local/lib/python3.11/dist-packages (from torch->pgmpy) (12.1.105)
Requirement already satisfied: nvidia-cuda-cupti-cu12==12.1.105 in /usr/local/lib/python3.11/dist-packages (from torch->pgmpy) (12.1.105)
Requirement already satisfied: nvidia-cudnn-cu12==9.1.0.70 in /usr/local/lib/python3.11/dist-packages (from torch->pgmpy) (9.1.0.70)
Requirement already satisfied: nvidia-cublas-cu12==12.1.3.1 in /usr/local/lib/python3.11/dist-packages (from torch->pgmpy) (12.1.3.1)
Requirement already satisfied: nvidia-cufft-cu12==11.0.2.54 in /usr/local/lib/python3.11/dist-packages (from torch->pgmpy) (11.0.2.54)
Requirement already satisfied: nvidia-curand-cu12==10.3.2.106 in /usr/local/lib/python3.11/dist-packages (from torch->pgmpy) (10.3.2.106)
Requirement already satisfied: nvidia-cusolver-cu12==11.4.5.107 in /usr/local/lib/python3.11/dist-packages (from torch->pgmpy) (11.4.5.107)
Requirement already satisfied: nvidia-cusparse-cu12==12.1.0.106 in /usr/local/lib/python3.11/dist-packages (from torch->pgmpy) (12.1.0.106)
Requirement already satisfied: nvidia-nccl-cu12==2.21.5 in /usr/local/lib/python3.11/dist-packages (from torch->pgmpy) (2.21.5)
Requirement already satisfied: nvidia-nvtx-cu12==12.1.105 in /usr/local/lib/python3.11/dist-packages (from torch->pgmpy) (12.1.105)
Requirement already satisfied: triton==3.1.0 in /usr/local/lib/python3.11/dist-packages (from torch->pgmpy) (3.1.0)
Requirement already satisfied: sympy==1.13.1 in /usr/local/lib/python3.11/dist-packages (from torch->pgmpy) (1.13.1)
Requirement already satisfied: nvidia-nvjitlink-cu12 in /usr/local/lib/python3.11/dist-packages (from nvidia-cusolver-cu12==11.4.5.107->torch->pgmpy) (12.6.85)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.11/dist-packages (from sympy==1.13.1->torch->pgmpy) (1.3.0)
Requirement already satisfied: googleapis-common-protos<2.0.dev0,>=1.56.2 in /usr/local/lib/python3.11/dist-packages (from google-api-core->google-generativeai->pgmpy) (1.66.0)
Requirement already satisfied: requests<3.0.0.dev0,>=2.18.0 in /usr/local/lib/python3.11/dist-packages (from google-api-core->google-generativeai->pgmpy) (2.32.3)
Requirement already satisfied: cachetools<6.0,>=2.0.0 in /usr/local/lib/python3.11/dist-packages (from google-auth>=2.15.0->google-generativeai->pgmpy) (5.5.1)
Requirement already satisfied: pyasn1-modules>=0.2.1 in /usr/local/lib/python3.11/dist-packages (from google-auth>=2.15.0->google-generativeai->pgmpy) (0.4.1)
Requirement already satisfied: rsa<5,>=3.1.4 in /usr/local/lib/python3.11/dist-packages (from google-auth>=2.15.0->google-generativeai->pgmpy) (4.9)
Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.11/dist-packages (from python-dateutil>=2.8.2->pandas->pgmpy) (1.17.0)
Requirement already satisfied: httplib2<1.dev0,>=0.19.0 in /usr/local/lib/python3.11/dist-packages (from google-api-python-client->google-generativeai->pgmpy) (0.22.0)
Requirement already satisfied: google-auth-httplib2<1.0.0,>=0.2.0 in /usr/local/lib/python3.11/dist-packages (from google-api-python-client->google-generativeai->pgmpy) (0.2.0)
Requirement already satisfied: uritemplate<5,>=3.0.1 in /usr/local/lib/python3.11/dist-packages (from google-api-python-client->google-generativeai->pgmpy) (4.1.1)
Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.11/dist-packages (from jinja2->torch->pgmpy) (3.0.2)
Requirement already satisfied: annotated-types>=0.6.0 in /usr/local/lib/python3.11/dist-packages (from pydantic->google-generativeai->pgmpy) (0.7.0)
Requirement already satisfied: pydantic-core==2.27.2 in /usr/local/lib/python3.11/dist-packages (from pydantic->google-generativeai->pgmpy) (2.27.2)
Requirement already satisfied: grpcio<2.0dev,>=1.33.2 in /usr/local/lib/python3.11/dist-packages (from google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.10.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,!=2.8.*,!=2.9.*,<3.0.0dev,>=1.34.1->google-ai-generativelanguage==0.6.15->google-generativeai->pgmpy) (1.69.0)
Requirement already satisfied: grpcio-status<2.0.dev0,>=1.33.2 in /usr/local/lib/python3.11/dist-packages (from google-api-core[grpc]!=2.0.*,!=2.1.*,!=2.10.*,!=2.2.*,!=2.3.*,!=2.4.*,!=2.5.*,!=2.6.*,!=2.7.*,!=2.8.*,!=2.9.*,<3.0.0dev,>=1.34.1->google-ai-generativelanguage==0.6.15->google-generativeai->pgmpy) (1.62.3)
Requirement already satisfied: pyasn1<0.7.0,>=0.4.6 in /usr/local/lib/python3.11/dist-packages (from pyasn1-modules>=0.2.1->google-auth>=2.15.0->google-generativeai->pgmpy) (0.6.1)
Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.11/dist-packages (from requests<3.0.0.dev0,>=2.18.0->google-api-core->google-generativeai->pgmpy) (3.4.1)
Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.11/dist-packages (from requests<3.0.0.dev0,>=2.18.0->google-api-core->google-generativeai->pgmpy) (3.10)
Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.11/dist-packages (from requests<3.0.0.dev0,>=2.18.0->google-api-core->google-generativeai->pgmpy) (2.3.0)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.11/dist-packages (from requests<3.0.0.dev0,>=2.18.0->google-api-core->google-generativeai->pgmpy) (2024.12.14)
Downloading pgmpy-0.1.26-py3-none-any.whl (2.0 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.0/2.0 MB 17.9 MB/s eta 0:00:00
Installing collected packages: pgmpy
Successfully installed pgmpy-0.1.26

import pandas as pd
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination

from google.colab import files
uploaded = files.upload()

import io
data = pd.read_csv(io.BytesIO(uploaded['heartdisease.csv']))

heart_disease = pd.DataFrame(data)
print(heart_disease)

    age  Gender  Family  diet  Lifestyle  cholestrol  heartdisease
0     0       0       1     1          3           0             1
1     0       1       1     1          3           0             1
2     1       0       0     0          2           1             1
3     4       0       1     1          3           2             0
4     3       1       1     0          0           2             0
5     2       0       1     1          1           0             1
6     4       0       1     0          2           0             1
7     0       0       1     1          3           0             1
8     3       1       1     0          0           2             0
9     1       1       0     0          0           2             1
10    4       1       0     1          2           0             1
11    4       0       1     1          3           2             0
12    2       1       0     0          0           0             0
13    2       0       1     1          1           0             1
14    3       1       1     0          0           1             0
15    0       0       1     0          0           2             1
16    1       1       0     1          2           1             1
17    3       1       1     1          0           1             0
18    4       0       1     1          3           2             0

model = BayesianNetwork([
    ('age', 'Lifestyle'),
    ('Gender', 'Lifestyle'),
    ('Family', 'heartdisease'),
    ('diet', 'cholestrol'),
    ('Lifestyle', 'diet'),
    ('cholestrol', 'heartdisease'),
    ('diet', 'cholestrol')
])

model.fit(heart_disease, estimator=MaximumLikelihoodEstimator)

HeartDisease_infer = VariableElimination(model)

print('For Age enter SuperSeniorCitizen:0, SeniorCitizen:1, MiddleAged:2, Youth:3, Teen:4')
print('For Gender enter Male:0, Female:1')
print('For Family History enter Yes:1, No:0')
print('For Diet enter High:0, Medium:1')
print('for LifeStyle enter Athlete:0, Active:1, Moderate:2, Sedentary:3')
print('for Cholesterol enter High:0, BorderLine:1, Normal:2')

For Age enter SuperSeniorCitizen:0, SeniorCitizen:1, MiddleAged:2, Youth:3, Teen:4
For Gender enter Male:0, Female:1
For Family History enter Yes:1, No:0
For Diet enter High:0, Medium:1
for LifeStyle enter Athlete:0, Active:1, Moderate:2, Sedentary:3
for Cholesterol enter High:0, BorderLine:1, Normal:2

Enter Age: 1
Enter Gender: 0
Enter Family History: 0
Enter Diet: 1
Enter Lifestyle: 3
Enter Cholestrol: 1

+-----------------+---------------------+
| heartdisease    |   phi(heartdisease) |
+=================+=====================+
| heartdisease(0) |              0.0000 |
+-----------------+---------------------+
| heartdisease(1) |              1.0000 |
+-----------------+---------------------+



Colab paid products - Cancel contracts here
++++++++++++++++++++++++++

ML exp 8_

Aim:Apply EM algorithm to cluster a set of data stored in a .CSV file. Use the same data set for clustering using k-Means algorithm. Compare the results of these two algorithms and comment on the quality of clustering. You can add Java/Python ML library classes/API in the program.

import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from google.colab import files
uploaded = files.upload()

Silhouette Score - EM Algorithm: 0.8480303059596955
Silhouette Score - k-Means Algorithm: 0.8480303059596955


Colab paid products - Cancel contracts here
++++++++++++++++++++

ML exp 8_

Aim:Apply EM algorithm to cluster a set of data stored in a .CSV file. Use the same data set for clustering using k-Means algorithm. Compare the results of these two algorithms and comment on the quality of clustering. You can add Java/Python ML library classes/API in the program.

import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from google.colab import files
uploaded = files.upload()

Silhouette Score - EM Algorithm: 0.8480303059596955
Silhouette Score - k-Means Algorithm: 0.8480303059596955


Colab paid products - Cancel contracts here
+++++++++++++++++++++++

ML exp 8_

Aim:Apply EM algorithm to cluster a set of data stored in a .CSV file. Use the same data set for clustering using k-Means algorithm. Compare the results of these two algorithms and comment on the quality of clustering. You can add Java/Python ML library classes/API in the program.

import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from google.colab import files
uploaded = files.upload()

Silhouette Score - EM Algorithm: 0.8480303059596955
Silhouette Score - k-Means Algorithm: 0.8480303059596955


Colab paid products - Cancel contracts here











