from flask import Flask,jsonify,request
import json
#Librerias para procesamiento de datos
# import re, string, unicodedata
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
model=joblib.load('SVM_model.pkl')

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
    predicted_pro = model.predict_proba(text_vectorized)
    predicted =''.join(str(e) for e in predicted)
    predicted_pro =''.join(str(e) for e in predicted_pro)
    if predicted == '1':
        return 'Real: '+predicted_pro
    else:
        return 'Fake: '+predicted_pro
    


if __name__ == '__main__':
    app.run()