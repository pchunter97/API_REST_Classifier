from flask import Flask,jsonify,request
import json
#Librerias para procesamiento de datos
import re, string, unicodedata
import nltk
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.lancaster import LancasterStemmer
#nltk.download('omv-1.4')
import inflect
from bs4 import BeautifulSoup
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer

import spacy
import es_core_news_lg
from spacy.lang.es.examples import sentences 

nlp = es_core_news_lg.load()

app = Flask(__name__)



#Ruta de prueba
@app.route('/classifier/<string:text>', methods=['GET'])
def ping(text):
    if len(text)>0:
        text=text.lower() #Convertir a minusculas
        text_tokenized = word_tokenize(text) #Tokenizar

        return jsonify({'text':text,'text_tokenized':text_tokenized})
    else:
        return jsonify({'text':'No text'})

@app.route('/get_classification', methods=['POST'])
def process_json():
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        json = request.json
        texto = json['texto']
        ##Preprocesamiento
        if len(texto)>0:
            #texto=texto.lower() #Convertir a minusculas
            #text_tokenized = word_tokenize(texto) #Tokenizar
            text_cleaned = data_cleaner(texto)
        ####################################
            return jsonify({'texto':texto,'text_tokenized':text_cleaned})
        else:
            return jsonify({'text':'No text'})
    else:
        return 'Content-Type not supported!'      

def data_cleaner(texto):
    texto = texto.lower() #Convertir a minusculas
    #Remover caracteres especiales
    texto = re.sub(r'[^\w\s]','',texto)
    #Tokenizar con spacy
    texto_tokenized = nlp(texto)
    #Remover stopwords
    texto = [word for word in texto.split() if word not in stopwords.words('spanish')]
    #Lematizar
    #lemmatizer = WordNetLemmatizer()
    #texto = [lemmatizer.lemmatize(word) for word in texto]
    #Stemming
    stemmer = LancasterStemmer()
    texto = [stemmer.stem(word) for word in texto]
    #print(texto)
    return texto_tokenized

if __name__ == '__main__':
    app.run(debug=True, port=4000)