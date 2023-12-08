
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
from ipywidgets import interact, widgets
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder



dataset_path = 'dataset'
num_classes = len(os.listdir(dataset_path))

#Fonction qui va être appelé pour effectuer la prediction
def effectuer_predictions(model, test_data, test_labels_encoded):
    # Effectuer les prédictions sur l'ensemble de test
    predictions = model.predict(test_data)
    predicted_classes = np.argmax(predictions, axis=1)
    # Créer un dictionnaire pour mapper les valeurs prédites aux noms de classes
    class_mapping = {0: "COVID", 1: "NORMAL", 2: "PNEUMONIA"}

    # Afficher la classe prédite et la classe réelle pour chaque image de l'ensemble de test
    for i in range(len(test_data)):
        predicted_class = class_mapping[predicted_classes[i]]
        true_class = class_mapping[test_labels_encoded[i]]

        print(f"Image {i+1}: Classe réelle - {true_class}, Classe prédite - {predicted_class}")

#Construction du modele
# Initialisation du modèle séquentiel
model = Sequential()
img_width=256
img_height=256
channels=3
# Première couche convolutive avec 32 filtres de taille 3x3 et fonction d'activation ReLU
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, channels)))
# Caractéristiques extraites : bords, contours, textures élémentaires

# Première couche de pooling avec une fenêtre de pooling de taille 2x2
model.add(MaxPooling2D((2, 2)))
# Réduction de la dimensionnalité des caractéristiques extraites

# Deuxième couche convolutive avec 64 filtres de taille 3x3 et fonction d'activation ReLU
model.add(Conv2D(64, (3, 3), activation='relu'))
# Caractéristiques extraites : motifs détaillés, structures anatomiques spécifiques

# Deuxième couche de pooling avec une fenêtre de pooling de taille 2x2
model.add(MaxPooling2D((2, 2)))
# Réduction supplémentaire de la dimensionnalité des caractéristiques extraites

# Troisième couche convolutive avec 128 filtres de taille 3x3 et fonction d'activation ReLU
model.add(Conv2D(128, (3, 3), activation='relu'))
# Caractéristiques extraites : caractéristiques plus abstraites, motifs plus complexes

# Troisième couche de pooling avec une fenêtre de pooling de taille 2x2
model.add(MaxPooling2D((2, 2)))
# Réduction finale de la dimensionnalité des caractéristiques extraites

# Couche Flatten pour aplatir les données en vue de les passer aux couches fully connected
model.add(Flatten())

# Première couche fully connected (Dense) avec 128 neurones et fonction d'activation ReLU
model.add(Dense(128, activation='relu'))
# Création de représentations plus abstraites des caractéristiques extraites

# Couche de sortie (Dense) avec un nombre de neurones égal au nombre de classes et fonction d'activation softmax
model.add(Dense(num_classes, activation='softmax'))
# Classification finale des images dans différentes classes (pneumonie, normal, covid)

# Charger les images et leurs étiquettes
images = []
labels = []

classes = os.listdir(dataset_path)
for class_name in classes:
    class_path = os.path.join(dataset_path, class_name)
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (256, 256))
        images.append(img)
        labels.append(class_name)


images = np.array(images)
labels = np.array(labels)

# Diviser les données en ensembles d'entraînement (80%) et de test (20%)
from sklearn.model_selection import train_test_split
train_data, test_data, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)


# Affichage du nombre d'images dans les ensembles d'entraînement et de test
print(f"Nombre d'images dans l'ensemble d'entraînement : {len(train_data)}")
print(f"Nombre d'images dans l'ensemble de test : {len(test_data)}")


# Prétraitement des étiquettes (encodage en valeurs numériques)
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
test_labels_encoded = label_encoder.transform(test_labels)

# Normalisation des données
train_data = train_data.astype('float32') / 255.0
test_data = test_data.astype('float32') / 255.0

# Compilation du modèle
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entraînement du modèle
history = model.fit(train_data, train_labels_encoded, epochs=10, batch_size=32, validation_data=(test_data, test_labels_encoded))
model.save('my_model')




# Appel de la fonction effectuer_predictions avec les arguments nécessaires
effectuer_predictions(model, test_data, test_labels_encoded)
# Calcul de la matrice de confusion
predictions = model.predict(test_data)
predicted_classes = np.argmax(predictions, axis=1)
confusion = confusion_matrix(test_labels_encoded, predicted_classes)
# Calcul de l'accuracy
accuracy = accuracy_score(test_labels_encoded, predicted_classes)

# Affichage de la matrice de confusion et de l'accuracy
print("Matrice de confusion :")
print(confusion)
print("\nAccuracy :", accuracy)

