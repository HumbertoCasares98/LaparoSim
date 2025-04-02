from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
from keras.models import load_model
import numpy as np

class Suture:
    def __init__(self):
        self.level = ["Experto", "Intermedio", "Novato"]
        self.modelTransf = load_model('Models/modeloSutura_74.h5')
        self.data = np.load('Datasets/Sutura/MapsData_Sutura.npy') # MAPS
        self.target = np.load('Datasets/Sutura/ClassData_Sutura.npy') # Clases (0: experto, 1: intermedio, 2: novato)
        self.target = np.array(self.target,dtype='int')
        self.y = self.target
        # Convertir las etiquetas en representación one-hot
        self.target = to_categorical(self.target)

        # Dividir los datos en conjuntos de entrenamiento, validación y prueba
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data, self.target, test_size=0.2, random_state=42)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=0.2, random_state=42)

        # Escalar los datos
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_val_scaled = self.scaler.transform(self.X_val)
        self.X_test_scaled = self.scaler.transform(self.X_test)

    def classify(self, maps):
        # Preprocess the new sample
        fer_scaled = self.scaler.transform(np.array([maps]))  # Assuming 'scaler' is the StandardScaler used for training
        # Predict the class probabilities for the new sample
        probabilities = self.modelTransf.predict(fer_scaled)
        # Get the predicted class label
        predicted_class = np.argmax(probabilities)

        # return the predicted class label
        return self.level[predicted_class]

