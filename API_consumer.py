import requests
import statistics
url = "https://fakenewsclassifier.azurewebsites.net/get_classification"
# url = "https://fake-classifier-news.herokuapp.com/get_classification"
tiempos=[]

def request():
    request = requests.post(url, json={'texto':'Gran sorpresa se llevó la hija de Donald Trump, la socialité y empresaria Ivanka Trump, luego de que una persona le enviara solicitud y cientos de mensajes a su cuenta ultra personal de Facebook, la cual solo se la da a personas conocidas. La mujer señala que por seguridad tiene un perfil secreto de Facebook con un nombre falso, el cual usa para comunicarse con sus amistades y familiares cercanos, por lo que le pareció raro que alguien le mandara solicitud y le escribiera sabiendo que es ella. Según varios medios, fue el hijo menor del nuevo presidente, López Obrador, el preadolescente Jesús López, quien logró dar con su cuenta con la finalidad de estar en contacto y conocerse más, esto luego de que el chico conociera por primera vez lo que es el amor al quedar impactado con la hermosura de Ivanka en la toma de protesta de su padre: Como todos los adolescentes de su edad, Chuyito esta conociendo lo que es el amor por primera vez, y era obvio que una mujer tan impactante como la hija de Trump le llamaría la atención. No podía creer que veía a alguien tan bella, por el trabajo de su papá Jesús solo había visto mujeres como Josefina Vázquez Mota, Delfina Gómez o Elba Esther Gordillo, pensó que todas las mujeres eran así y por eso cuando vio a Ivanka fue como si viera a un ángel, fue amor a primera vista, comentó un reportero de la revista ¡Hola!, que ahora se dedicará a subir puras notas de Obrador y su familia para haber si agarran hueso. El periodista señala que Chuy logró dar con el Facebook personal de Ivanka, lu ego de que esta se lo diera a la nueva Primera Dama de México, Beatriz Gutiérrez: Trump es muy astuta y le dio a Beatriz su Facebook personal supuestamente para generar empatía pero todo se trata de una estrategia para tratar de manipularla a ella y Obrador. Lo que no contó es que Chuyito revisaría el Facebook de su mamá para dar con el contacto de ella. Tras revisar la red social de su madre sin permiso, el chico finalmente encontró el Facebook de Ivanka, porque anteriormente ya le había escrito a su fan page pero no le respondí a, así que con la cuenta personal no tendría pretextos para que su mensaje no le llegara, dijo. Según comenta el periodista, durante la toma de protesta Jesús no cruzó palabra con Trump, por eso en sus mensajes le recordó que era el chico que estaba sentado a su lado: Hola Ivanka espero estés bien ¿sabes quien soy?, soy el hijo del nuevo presidente de México, el que estaba a tu lado. Te quería saludar pero me dio pena, la verdad te me hiciste buena onda y quiero ver si me aceptas y podemos hablar, y un día salir a tomar algo. Mi papá me da $10 mil pesos para gastar cada domingo, te puedo invitar al Starbucks o a Six Flags, o igual lo convenzo de yo ir a Estados Unidos y vernos ¿te parece?, por cierto, muy guapa en tus fotos, te daré like a todas, y me llamo Jesús, pero me puedes decir Chuy, te quiero mucho, le escribió el niño. Según varios medios, Ivanka simplemente respondió el mensaje de Chuy con un icono de manita arriba, y aún no ha aceptado la solicitud aunque tampoco la ha rechazado, para no causar un conflicto entre México y Estados Unidos.'})
    if request.status_code == 200:
        print(request.json())
        data = request.json()
        print(data['texto_clasificado'])
        print("Tiempo de respuesta: " + str(request.elapsed))
        #Convertir el tiempo de respuesta a milisegundos
        tiempos.append(request.elapsed.total_seconds())

        # tiempos.append(request.elapsed)
        
            

for i in range(10):
    print("Resultado: " + str(i))
    request()
    print("\n")

print("Tiempos de respuesta: " + str(tiempos))
# print("Promedio de tiempo de respuesta: " + str(sum(tiempos)/len(tiempos)))
print("Tiempo de respuesta máximo: " + str(max(tiempos)))	
print("Tiempo de respuesta mínimo: " + str(min(tiempos)))
print("Tiempo de respuesta medio: " + str(sum(tiempos)/len(tiempos)))
# print("Tiempo de respuesta desviación estándar: " + str(round(statistics.stdev(tiempos),2)))
# print("Tiempo de respuesta media aritmética: " + str(round(statistics.mean(tiempos),2)))
# print("Tiempo de respuesta mediana: " + str(round(statistics.median(tiempos),2)))
# print("Tiempo de respuesta moda: " + str(round(statistics.mode(tiempos),2)))
# print("Tiempo de respuesta varianza: " + str(round(statistics.variance(tiempos),2)))
