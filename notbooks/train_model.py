import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("data/patients_dakar.csv")
print(f"Dataset : {df.shape[0]} patients, {df.shape[1]} colonnes")
print(f"Colonnes : {list(df.columns)}")
print(f"Diagnostics : {df['diagnostic'].value_counts()}")

le_sexe = LabelEncoder()
le_region = LabelEncoder()
df['sexe_encoded'] = le_sexe.fit_transform(df['sexe'])
df['region_encoded'] = le_region.fit_transform(df['region'])

feature_cols = ['age', 'sexe_encoded', 'temperature', 'tension_sys',
                'toux', 'fatigue', 'maux_tete', 'region_encoded']
X = df[feature_cols]
y = df['diagnostic']

print(f"Features : {X.shape}")
print(f"Cible : {y.shape}")
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Entrainement : {X_train.shape[0]} patients")
print(f"Test : {X_test.shape[0]} patients")
from sklearn.metrics import confusion_matrix, classification_report

y_pred = model.predict(X_test)

print("\nRapport de classification :")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
print("Matrice de confusion :")
print(cm)
model_loaded = joblib.load("models/model.pkl")
le_sexe_loaded = joblib.load("models/encoder_sexe.pkl")
le_region_loaded = joblib.load("models/encoder_region.pkl")

print(f"Modele recharge : {type(model_loaded).__name__}")
print(f"Classes : {list(model_loaded.classes_)}")
# Un nouveau patient arrive au centre de sante de Medina
nouveau_patient = {
'age ': 28,
'sexe ': 'F',
' temperature ': 39.5 ,
' tension_sys ': 110 ,
'toux ': True ,
'fatigue ': True ,
'maux_tete ': True ,
'region ': 'Dakar '
}
# Encoder les valeurs categoriques
sexe_enc = le_sexe_loaded . transform ([ nouveau_patient ['sexe ']]) [0]
region_enc = le_region_loaded . transform ([ nouveau_patient ['region ']]) [0]
# Preparer le vecteur de features
features = [
nouveau_patient ['age '],
sexe_enc ,
nouveau_patient [' temperature '],
nouveau_patient [' tension_sys '],
int ( nouveau_patient ['toux ']) ,
int ( nouveau_patient ['fatigue ']) ,
int ( nouveau_patient ['maux_tete ']) ,
region_enc
]
# Predire
diagnostic = model_loaded . predict ([ features ]) [0]
probas = model_loaded . predict_proba ([ features ]) [0]
proba_max = probas . max ()
print (f"\n--- Resultat du pre - diagnostic ---")
print (f" Patient : { nouveau_patient [' sexe ']}, { nouveau_patient [' age ']} ans
")
print (f" Diagnostic : { diagnostic }")
print (f" Probabilite : { proba_max :.1%} ")
print (f"\ nProbabilites par classe :")
for classe , proba in zip ( model_loaded . classes_ , probas ):
bar = '#' * int ( proba * 30)
print (f" { classe :8s} : { proba :.1%} { bar }")
# Simuler ce que fera l'API en Lab 3 :
# Charger le modele DEPUIS LE FICHIER ( pas depuis la memoire )
model_loaded = joblib . load (" models / model . pkl ")
le_sexe_loaded = joblib . load (" models / encoder_sexe . pkl ")
le_region_loaded = joblib . load (" models / encoder_region . pkl ")
print (f" Modele recharge : { type ( model_loaded ). __name__ }")
print (f" Classes : { list ( model_loaded . classes_ )}")
# Un nouveau patient arrive au centre de sante de Medina
nouveau_patient = {
'age ': 28,
'sexe ': 'F',
' temperature ': 39.5 ,
' tension_sys ': 110 ,
'toux ': True ,
'fatigue ': True ,
'maux_tete ': True ,
'region ': 'Dakar '
}
# Encoder les valeurs categoriques
sexe_enc = le_sexe_loaded . transform ([ nouveau_patient ['sexe ']]) [0]
region_enc = le_region_loaded . transform ([ nouveau_patient ['region ']]) [0]
# Preparer le vecteur de features
features = [
nouveau_patient ['age '],
sexe_enc ,
nouveau_patient [' temperature '],
nouveau_patient [' tension_sys '],
int ( nouveau_patient ['toux ']) ,
int ( nouveau_patient ['fatigue ']) ,
int ( nouveau_patient ['maux_tete ']) ,
region_enc
]
# Predire
diagnostic = model_loaded . predict ([ features ]) [0]
probas = model_loaded . predict_proba ([ features ]) [0]
proba_max = probas . max ()
print (f"\n--- Resultat du pre - diagnostic ---")
print (f" Patient : { nouveau_patient [' sexe ']}, { nouveau_patient [' age ']} ans
")
print (f" Diagnostic : { diagnostic }")
print (f" Probabilite : { proba_max :.1%} ")
print (f"\ nProbabilites par classe :")
for classe , proba in zip ( model_loaded . classes_ , probas ):
bar = '#' * int ( proba * 30)
print (f" { classe :8s} : { proba :.1%} { bar }")