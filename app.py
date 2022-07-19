from flask import Flask,jsonify,request, render_template
import json
#Librerias para procesamiento de datos
import keras
import tensorflow as tf
import re, string, unicodedata
# import nltk
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')
# from nltk.stem.wordnet import WordNetLemmatizer
# from nltk.stem.lancaster import LancasterStemmer
#nltk.download('omv-1.4')
# import inflect
# from bs4 import BeautifulSoup
# from nltk import word_tokenize, sent_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import LancasterStemmer, WordNetLemmatizer

import spacy
import es_core_news_lg
from spacy.lang.es.examples import sentences 

import joblib
import numpy as np

nlp = es_core_news_lg.load()
# model=joblib.load('SVM_model.pkl')
from keras.models import load_model
#####F1
from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
dependencies = {
    'f1_m': f1_m,
    'precision_m': precision_m,
    'recall_m': recall_m,
}
# model = keras.models.load_model('LSTM_training.h5')
model = load_model("LSTM_training.h5",custom_objects=dependencies)



app = Flask(__name__)

#Ruta de prueba despliegue
@app.route('/')
def index():
    return 'Hello, World!'
    

#Ruta de prueba
@app.route('/classifier/<string:text>', methods=['GET'])
def ping(text):
    if len(text)>0:
        text=text.lower() #Convertir a minusculas
        text_tokenized = word_tokenize(text) #Tokenizar

        return jsonify({'text':text,'text_tokenized':text_tokenized})
    else:
        return jsonify({'text':'No text'})

#Ruta de clasificación
@app.route('/get_classification', methods=['POST'])
def process_json():
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        json = request.json
        texto = json['texto']
        ##Preprocesamiento
        if len(texto)>0:
        ####################################
        ##Clasificación
            texto_clasificado= clasificar(texto)
        ####################################
            #return jsonify({'texto':texto,'text_tokenized':text_cleaned})
            return jsonify({'texto_clasificado':texto_clasificado})

        else:
            return jsonify({'text':'No text'})
    else:
        return 'Content-Type not supported!'      

#Retornar página con peticion get and return index.html file
@app.route('/get_classification', methods=['GET'])
def get_page():
    #return app.send_static_file('index.html')
    #return the index.html file
    return render_template('index.html')
    # return 'Hello, World!'



def preprocesar(texto):
    texto = texto.lower() #Convertir a minusculas
    #Remover caracteres especiales
    texto = re.sub(r'[^\w\s]','',texto)
    #Tokenizar con spacy
    doc = nlp(texto)
    #Limpieza de datos
    tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-' and not tok.is_punct and not tok.is_stop]
    texto = ' '.join(tokens)
    text_vectorized = []

    nlp.pipe(texto)
    if doc.has_vector:
        text_vectorized.append(doc.vector)
        print(doc.text, doc.vector)
    else:
        text_vectorized.append(np.zeros((128,), dtype="float32"))

    return text_vectorized

def clasificar(texto):
    #Array de ejemplos
    text_vectorized = np.array(preprocesar(texto))

    predicted = model.predict(text_vectorized)
    #predicted_pro = model.predict_proba(text_vectorized)
    #predicted =''.join(str(e) for e in predicted)
    #predicted_pro =''.join(str(e) for e in predicted_pro)
    if predicted > 0.5:
        return 'Real: '+str(predicted)
    else:
        return 'Fake: '+str(predicted)
    


if __name__ == '__main__':
    app.run(debug=True)