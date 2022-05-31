import nltk
import es_dep_news_trf

from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

import pandas

####
nlp = es_dep_news_trf.load()
doc = nlp("El año pasado la Ciudad de México se estremeció luego de que se anunciara que Godzilla llegaría a la ciudad , pero no de forma literal porque el mounstro ni siquiera existe en la vida real, sino que los productores filmarían escenas de la nueva película lo que originó una gran expectativa y sobre todo, divertidos chistes y memes sobre el fatal destino que tendría Godzilla al pisar nuestro país. Fue poco lo que se supo del tema, principalmente porque el rodaje fue más rápido de lo que se esperaba (solo dos semanas), y a diferencias de otras cintas como la de James Bond 'Spectre' donde se denunció que habían dañado edificios, la filmación de Godzilla corrió sin contratiempos, como si solo hubieran grabado tomas a las fachadas de algunos edificios. Como estas películas tardan años en salir a la luz, finalmente mostraron dos trailers oficiales de la cinta que se estrenará en el verano del 2019, y veremos como Godzilla destruye la CDMX, o por lo menos, la ciudad aparecerá unos segundos en la cinta ya que muchas veces estas cintas filmadas aquí hacen mucho ruido, pero en realidad nuestro país termina solo saliendo un par de minutos a cuadro. ")

#print([(w.text, w.pos_) for w in doc])
#for token in doc:
#    print(token) 
#Eliminate stopwords and punctuation
# lexical_tokens = [t.orth_ for t in doc if not t.is_punct | t.is_stop]
# print(type(doc))
#Lematizador: Volver a la forma base a la palabra
#Eliminar stopwords, puntuación y lematizar
tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-' and not tok.is_punct and not tok.is_stop]


#Select stop words
for word in doc:
    if word.is_stop == True:
        print(word)

print(lexical_tokens)
#print("Lemas: ", lemmas)
print("\n\nTokens: ", tokens)
###NLTK
# print("-------NLTK-----")
# texto = "Esto es una frase."	
# texto = [word for word in texto.split() if word not in stopwords.words('spanish')]
# print(texto)
