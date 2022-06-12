
#import es_dep_news_trf
#import es_core_news_lg
import es_core_news_md
from spacy import displacy

import pandas as pd
import numpy as np

####
#nlp = es_dep_news_trf.load()
#doc = nlp("El año pasado la Ciudad de México se estremeció luego de que se anunciara que Godzilla llegaría a la ciudad , pero no de forma literal porque el mounstro ni siquiera existe en la vida real, sino que los productores filmarían escenas de la nueva película lo que originó una gran expectativa y sobre todo, divertidos chistes y memes sobre el fatal destino que tendría Godzilla al pisar nuestro país. Fue poco lo que se supo del tema, principalmente porque el rodaje fue más rápido de lo que se esperaba (solo dos semanas), y a diferencias de otras cintas como la de James Bond 'Spectre' donde se denunció que habían dañado edificios, la filmación de Godzilla corrió sin contratiempos, como si solo hubieran grabado tomas a las fachadas de algunos edificios. Como estas películas tardan años en salir a la luz, finalmente mostraron dos trailers oficiales de la cinta que se estrenará en el verano del 2019, y veremos como Godzilla destruye la CDMX, o por lo menos, la ciudad aparecerá unos segundos en la cinta ya que muchas veces estas cintas filmadas aquí hacen mucho ruido, pero en realidad nuestro país termina solo saliendo un par de minutos a cuadro. ")
text="Este es una prueba de texto"
# nlp = es_core_news_lg.load()
nlp = es_core_news_md.load()
doc = nlp(text)

#displacy.render(doc,style='dep')

#Lematizador: Volver a la forma base a la palabra
#Eliminar stopwords, puntuación y lematizar
tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-' and not tok.is_punct and not tok.is_stop]
#save tokens in a string
text = ' '.join(tokens)
train_vec = []
print(text)

#save text in pandas dataframe
#df = pd.DataFrame(train_vec, columns=['text'])
#for doc in nlp.pipe(df,batch_size=500):
nlp.pipe(text,batch_size=50)
if doc.has_vector:
    train_vec.append(doc.vector)
    print(doc.text, doc.vector)
else:
    #print(doc.text, 'no vector')
    train_vec.append(np.zeros((128,), dtype="float32"))

print(len(train_vec[0]))
print(len(tokens))
# print(train_vec)
# for doc in 
#     print(doc.text)

#Vectorización feature engineering TF-IDF
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.pipeline import Pipeline
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# from sklearn.svm import LinearSVC

# tfidf = TfidfVectorizer(tokenizer = tokens)
# classifier = LinearSVC()

# clf = Pipeline([('tfidf', tfidf), ('clf', classifier)])

# print(tfidf)
# print(classifier)
# print(clf)
# X_train, X_test, y_train, y_test = train_test_split(tokens, test_size=0.2, random_state=42)
# clf.fit(X_train, y_train)
#Start a pd
#df = pd.DataFrame(text)
#Save 
#print(df)
#print(type(df))
# text_preprocesado = []
# for doc in nlp.pipe(df, batch_size=5000):
#     if doc.has_vector:
#         train_vec.append(doc.vector)
