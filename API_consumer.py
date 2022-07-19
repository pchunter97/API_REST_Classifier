import requests
import statistics
import pandas as pd 
from tqdm import tqdm


url = "https://fakenewsclassifier.azurewebsites.net/get_classification"
# url = "https://fake-classifier-news.herokuapp.com/get_classification"
tiempos=[]

def request(texto):
    request = requests.post(url, json={'texto':texto})
    if request.status_code == 200:
        print(request.json())
        data = request.json()
        print(data['texto_clasificado'])
        print("Tiempo de respuesta: " + str(request.elapsed))
        #Convertir el tiempo de respuesta a milisegundos
        tiempos.append(request.elapsed.total_seconds())

        # tiempos.append(request.elapsed)
        
#Obtener las noticias para probar la API
dataframe = pd.read_csv('C:/Users/Dell/Desktop/Trabajo de titulación/Dataset/dataset_final.csv')
short = 0
medium = 0
long = 0
for idx in tqdm(range(len(dataframe))):
    if(len(dataframe['texto'].iloc[idx])>1300 and len(dataframe['texto'].iloc[idx])<2000):
        short = short+1
        request(dataframe['texto'].iloc[idx])
    # elif(len(dataframe['texto'].iloc[idx])>=2000 and len(dataframe['texto'].iloc[idx])<3300):
    #     medium = medium+1
    # elif(len(dataframe['texto'].iloc[idx])>=3300):
    #     long = long+1
#############################################################################################################################
#############################################################################################################################

# for i in range(10):
#     print("Resultado: " + str(i))
#     request("xD")
#     print("\n")

print(f"Tiempos de respuesta de :{short} noticias " + str(tiempos))
# print("Promedio de tiempo de respuesta: " + str(sum(tiempos)/len(tiempos)))
print("Tiempo de respuesta máximo: " + str(max(tiempos)))	
print("Tiempo de respuesta mínimo: " + str(min(tiempos)))
print("Tiempo de respuesta medio: " + str(sum(tiempos)/len(tiempos)))
# print("Tiempo de respuesta desviación estándar: " + str(round(statistics.stdev(tiempos),2)))
# print("Tiempo de respuesta media aritmética: " + str(round(statistics.mean(tiempos),2)))
# print("Tiempo de respuesta mediana: " + str(round(statistics.median(tiempos),2)))
# print("Tiempo de respuesta moda: " + str(round(statistics.mode(tiempos),2)))
# print("Tiempo de respuesta varianza: " + str(round(statistics.variance(tiempos),2)))
