# Ball bounce detector
Esta aplicación consiste en un detector de cambios de dirección de una pelota de pádel a partir de un dataset obtenido en el que la pelota se ha etiquetado para cada frame. La pelota viene dada en cada frame por las coordenadas de su centro, "x" e "y" en píxeles siendo "x" el número de píxeles en horizontal desde la esquina superior izquierda de la imagen hacia la derecha y el eje "y" el número de píxeles desde la esquina superior izquierda en vertical hacia abajo.

El dataset de origen se encuentra en el directorio "datasets/2022-master-finals-fem" y contiene 2 archivos:

- El dataset original con todos los datos anotados "datasets/2022-master-finals-fem/2022-master-finals-fem.csv"
- Un archivo de muestra creado manualmente para validar el resultado del detector para una serie de frames.

Los archivos tienen la siguiente estructura.

## Arhivos dataset

### Dataset original

Archivo situado en "datasets/2022-master-finals-fem/2022-master-finals-fem.csv" con las siguientes columnas (marco solo las relevantes para la tarea en cuestión):

- Número de fila -1: número de frame
- has_shot: si algún jugador está efectuando un golpe sobre la pelota. Abarca desde la preparación del golpe hasta el acabado y el contacto entre la pala y la pelota se suele producir en algún frame intermedio.
- category: la categoría del golpe. Puede ser "Serve", "Forehand", "Backhand", "Smash" y "Other".
- ball_center_x: coordenada "x" en píxeles del centro de la bola.
- ball_center_y: coordenada "y" en píxeles del centro de la bola.
- serving: iniciales del jugador que está sacando en ese juego.
- serving_team: el equipo que está sacando: "a" o "b"
- upper_team: el equipo que está jugando arriba en la imagen o en la zona "más alejada" de la cámara.
- lower_team: el equipo que está jugando abajo en la imagen o en la zona "más cercana" de la cámara.
- team_shot: el equipo que está efectuando el golpe si has_shot vale 1.
- player_<a/b>_<left/drive>_<x/y/w/h>: bbox de cada jugador (equipo a o b, posición left o drive, coordenada x, y, w y h de la bbox)
- time: el tiempo en segundos correspondiente al cada frame.
- shot_result: el resultado del golpeo efectuado en caso de que has_shot sea 1. Puede valer 1 si el golpeo es bueno y el juego continua, 0 si el golpeo es malo y se acaba el punto, 2 si el golpeo es winner y también se acaba el punto.


### Archivo de muestra de rebotes

Archivo situado en "datasets/2022-master-finals-fem/2022-master-finals-fem-bounces-sample.csv" que contiene en qué frames se 
producen rebotes de la pelota. Contiene una sola columna llamada "bounce_frame" e indica que en ese frame hay un rebote. Se considera rebote cualquier cambio de dirección que sufra la pelota tras impactar con cualquier superficie. Puede ser la propia pala de un jugador durante un golpeo, el rebote en el suelo, cristales o vallas. Hay que distinguir este tipo de movimiento del propio cambio de dirección sufrido por el efecto de la gravedad.


### Objetivo de la aplicación

El objetivo de la aplicación es maximizar una función objetivo, que he definido como:

FO = TP / (TP + FP + FN) donde:

TP: true positives -> rebote detectado en momento oportuno
FP: false positives -> rebote detectado cuando no hay rebote
FN: false negatives -> rebote real no detectado

De esta manera, la función valdrá 1 sólo cuando se detecten todos y únicamente los rebotes reales.


### Consideraciones

Las anotaciones de la posición de la pelota están hechas de manera manual, por lo que existe un ruido inerente a la trayectoria de la bola causada por pequeños errores a la hora de introducir los datos. Además, según la pelota vaya en una dirección u otra, el rebote se hace mucho más evidente o menos. Por ejemplo, si la pelota va en una dirección perpendicular a la propia pantalla, su centro casi se mantendrá sin movimiento y al rebotar puede que tampoco se note un cambio de dirección excesivo. Sin embargo, con el ojo humano se podría detectar más del 95% de los rebotes en cada momento por lo que yo creo que tiene que existir alguna manera de detectar con una precisión similar en qué frames la pelota rebota realmente.

Además, se puede aceptar un margen de frames en los que se consideraría un "TP", por ejemplo, si el rebote se detecta y está a 1 frame de distancia del rebote real se consideraría como un "TP".