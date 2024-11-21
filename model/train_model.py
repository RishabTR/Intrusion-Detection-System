import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv(r'A:/Project/IDS/model/filtered_ids1.csv')

data = data.replace([float('inf'), float('-inf')], float('nan'))
data.dropna(inplace=True)

label_encoder = LabelEncoder()
data['Label'] = label_encoder.fit_transform(data['Label'])

label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("Label Mapping:", label_mapping)

X = data.drop('Label', axis=1)  
y = data['Label']              

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")

with open('model/model.pkl', 'wb') as file:
    pickle.dump(model, file)

with open('model/label_mapping.pkl', 'wb') as file:
    pickle.dump(label_mapping, file)

