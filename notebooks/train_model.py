import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib, os

df = pd.read_csv("data/patients_dakar.csv")
print(f"Dataset : {df.shape[0]} patients, {df.shape[1]} colonnes")

le_sexe = LabelEncoder()
le_region = LabelEncoder()
df["sexe_encoded"] = le_sexe.fit_transform(df["sexe"].astype(str))
df["region_encoded"] = le_region.fit_transform(df["region"].astype(str))

feature_cols = ["age", "sexe_encoded", "temperature", "tension_sys", "toux", "fatigue", "maux_tete", "region_encoded"]
X = df[feature_cols]
y = df["diagnostic"]
print(f"Features : {X.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Entrainement : {X_train.shape[0]} patients")
print(f"Test : {X_test.shape[0]} patients")

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Modele entraine !")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy : {accuracy:.2%}")

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")
joblib.dump(le_sexe, "models/encoder_sexe.pkl")
joblib.dump(le_region, "models/encoder_region.pkl")
joblib.dump(feature_cols, "models/feature_cols.pkl")
print("Modele sauvegarde !")

model_loaded = joblib.load("models/model.pkl")
le_sexe_loaded = joblib.load("models/encoder_sexe.pkl")
le_region_loaded = joblib.load("models/encoder_region.pkl")
print(f"Modele recharge : {type(model_loaded).__name__}")
print(f"Classes : {list(model_loaded.classes_)}")

nouveau_patient = {"age": 28, "sexe": "F", "temperature": 39.5, "tension_sys": 110, "toux": 1, "fatigue": 1, "maux_tete": 1, "region": "Dakar"}
sexe_enc = le_sexe_loaded.transform([nouveau_patient["sexe"]])[0]
region_enc = le_region_loaded.transform([nouveau_patient["region"]])[0]
features = [nouveau_patient["age"], sexe_enc, nouveau_patient["temperature"], nouveau_patient["tension_sys"], nouveau_patient["toux"], nouveau_patient["fatigue"], nouveau_patient["maux_tete"], region_enc]
diagnostic = model_loaded.predict([features])[0]
probas = model_loaded.predict_proba([features])[0]
print(f"Diagnostic : {diagnostic}")
print(f"Probabilite : {probas.max():.1%}")
