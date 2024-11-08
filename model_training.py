import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


# Converting to DataFrame
#df = pd.DataFrame(data)
file_path = 'A_year_of_pizza_sales_from_a_pizza_place_872_68 (1).csv'
df = pd.read_csv(file_path)

# Drop unnecessary columns
df = df.drop(['Unnamed: 0', 'X', 'id'], axis=1)

# Convert the 'date' column to datetime format and extract month and day
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

# Extract the hour from the 'time' column and drop 'date', 'time', and 'day' columns
df['hour'] = df['time'].str.split(':').str[0].astype(int)
columns_to_remove = ['date', 'time', 'day']
df = df.drop(columns=columns_to_remove)




# Preprocessing (e.g., encoding categorical variables)

df['name'] = df['name'].astype('category').cat.codes
df['size'] = df['size'].astype('category').cat.codes
df['type'] = df['type'].astype('category').cat.codes
df['price'] = df['price'].astype('category').cat.codes
df['month'] = df['month'].astype('category').cat.codes

# Splitting into features and target
#X = df.drop(['type', 'id', 'date', 'time', 'name'], axis=1)
#y = df['type']
X_dt = df.drop(columns=['type'])  # Features except the target 'type'
y_dt = df['type']  # Target variable 'type'

# Splitting the dataset into training and test sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_dt, X_test_dt, y_train_dt, y_test_dt = train_test_split(X_dt, y_dt, test_size=0.2, random_state=42)

# Training a simple model
#model = RandomForestClassifier()
#model.fit(X_train, y_train)
model_dt = DecisionTreeClassifier(random_state=42)
model_dt.fit(X_train_dt, y_train_dt)


# Save the model to a file
with open('DecisionTreeClassifier.pkl', 'wb') as model_file:
    pickle.dump(model_dt, model_file)

print("Model saved as pizza_type_model.pkl")
