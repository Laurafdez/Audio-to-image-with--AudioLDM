# Audio-to-image
Este repositorio va a mostrar un modelo que utiliza AudioLDM para sonorizar imagenes. Para ello, se ha utilizado el modelo AudioLDM preentrenado para poder generar audios cuando recibe como entrada una imagen. Se han seguido distintos paso para llegar a dicha implementación. AudioLDM es un modelo que es capaz de generar audio a partir de un texto, para poder realizar el proceso cuenta en su implementación con el modelo CLAP. La función principal de CLAP es la de condicionar los audios generados en función de un texto a la entrada o lo que es lo mismo, los embeddings que salen del módulo CLAP son vectores que condicionan la generación de audio por el texto. AudioLDM utiliza el codificador de CLAP para codificicar sus textos. Este hecho se va a aprovechar para desarrollar una red neuronal Traductor que sea capaz de traducir los embeddings de CLIP en embeddings que se parezcan a embeddings de CLAP, controlando este proceso con la ayuda de una función de coste. Se busca traducir los embeddings de CLIP en embeddings de CLAP, los embeddings de las imágenes que codifica CLIP están condicionados por textos y los embeddings de los textos por las imágenes. Como el objetivo de este modelo es generar audio a partir de una imagen resulta conveniente usar CLIP para este proceso. CLIP es modelo multimodal creado por la empresa OpenAI que utiliza el aprendizaje zero shot y utiliza NLP para supervisar el entrenamiento, es decir, es capaz de trabajar con un dato que no ha visto antes solo conociendo su descripción o nombre. CLIP está formado por una red neuronal que ha sido entrenado por una variedad de pares(imagen, texto). Busca que dada una imagen es capaz de encontrar el mejor texto para esa imagen. Una vez que se tenga el modelo Traductor creado, se sustituirá el encoder de texto de AudioLDM por este modelo para conseguir que AudioLDM sea capaz de generar sonido dada una imagen.

En este repositprio se van a explicar los pasos seguidos para desarrollar este proceso.

## Fase I. Base de datos

El primer paso es sacar los embeddings de las distintas bases de datos que se han utlizado para el entrenamiento de los modelos translate. Primero de todo, se comienza a sacar los embeddings de la base de datos que utilizó AudioLDM para su entrenamiento, esta fue Audiocaps. Audiocaps es una base de datos que se encarga de describir en lenguaje natural cualquier tipo de audio en condiciones reales. Esta base de datos consta de 57000 pares de clips de audios y descripciones de texto escrito por humanos.

La columna que es importante para la implementación de la red neuronal es la de captions, ya que en ella se encuentran las descripciones de los sonidos. Una vez descargada toda la base de datos se procede a obtener los embeddings de la columna captions tanto por el codificador de CLIP como por el codificador de CLAP, ya que lo que se busca es desarrollar un modelo que sea capaz de traducir unos embeddings en otros.

Se realiza el proceso de codificación de cada una de las captions por medio del codificador de texto de CLIP y de CLAP. Se genera un archivo de texto en el que se recogen los embeddings de CLIP y de CLAP para una misma descripción de textos. Estos embeddings son vectores de 1x512. Es importante conocer si los valores se encuentran normalizados o no, por ello, se sacan los histogramas de las normas de ambos embeddings.

En la siguiente imagen se puede observar como los histogramas de CLIP no se encuentran normalizados, este es un dato muy importante porque a la hora de diseñar la red neuronal a de tenerse en cuenta, ya que la entrada del módulo deberá normalizar los valores antes de trabajar con ellos.



Por otro lado, para los embeddings de CLAP, observables en la siguiente imagen, sus normas sí que están normalizadas entre 0 y 1, es decir, en este caso no es necesario que a la salida de la red neuronal se haga un proceso de desnormalización.


Para poder reproducir el proceso de sacar los embeddings de las base de datos de Audiocaps, es necesario descargarla de la siguiente enlace https://audiocaps.github.io/ y sacar los siguientes pasos:


1. Clone the repository and navigate to where the code is:
   ```console
      git clone https://github.com/Laurafdez/Audio-to-image-with--AudioLDM.git
      ```
2. Navigate to where the code is:
   ```console
     cd /Audio-to-image-with--AudioLDM.git/Prior steps/audiocaps
      ```
       
2. En los siguientes scripts solo hay que cambiar el csv con el nombre de la base de datos Audiocaps y ejecutar
   ```console
     python embedding_CLIP.py
     ```
3. Y para sacar los embeddings de CLAP:
   ```console
     python embedding_CLAP.py
     ```
4. Se habrán creado dos csv con los embeddings de CLIP y CLAP, para crear archivos de textos con esa información de ejecuta el siguiente script:
   ```console
     python csvtotxt.py
     ```
5.  En este punto ya se tendrán todos los embeddings tanto de CLIP como los correspondientes de CLAP en archivos de texto para cada una de los datos de Adiocaps.
6.  Si se quieren sacar los histogramas de CLIP o CLAP habrá que cambiar el nombre del csv del archivo histogram.py y ejecutarlo:
   ```console
     cd; python histogram.py
     ```
Con estos datos se entrenaran 6 modelos de Translate con distintas funciones de coste y de capa intermedia: 256 o 512. Y las funciones de coste son: distancia coseno, error cuadrático medio y constractive learning.

Adicionalmente, como se busca generar audios lo más fieles posible a las imágenes se decide entrenar 12 modelos más, la diferencia de estos modelos es que se van a entrenar 6 modelos primeros cuya base de datos sea la misma que ha utilizado CLIP en su entrenamiento. De esta manera los nuevos modelos Translator van a trabajar con los embeddings directamente de las imágenes y los traducirán en embeddings parecidos a los embeddings de CLAP. Para poder llevar a cabo este proceso se han descargado dos nuevas bases de datos. Por un lado, la base de datos que utiliza CLIP para entrenar, WIT:conjunto de datos de texto de imagen basado en Wikipedia. WIT son datos de texto de imagen basado en Wikipedia, es un conjunto multilingüe multimodal que se compone por 37,6 millones de ejemplos de textos con 11,5 millones de imágenes únicas en 108 idiomas. Este conjunto de datos se creó extrayendo múltiples textos asociados con una imagen de los artículos de Wikipedia y los enlaces de las imágenes de Wikimedia. Es el conjunto de datos más grande que existe hasta la fecha. Se utiliza sobre todo para el preentrenamiento de los modelos multimodales y presenta una alta calidad de alineación entre las imágenes y los textos. Se implementa un script que se encarga de descargar cada una de esas imágenes y sacar los embeddings correspondientes mediante el codificador de CLIP. Asimismo, se codifican las descripciones correspondientes de esas imágenes y con la ayuda del codificador de CLAP. Cada uno de estos embeddings tantos los de CLIP como los de CLAP se guardan en archivos txt. En total, se generan 10.000 archivos con la información de estas imágenes. Como el objetivo es generar la misma cantidad de muestras que las que se obtuvieron con la base de datos Audiocaps, se decide descargar otra base de datos para completar las muestras. El problema de la base de datos de WIT es que muchos de los enlaces ya no estaban disponibles. Además, se busca que el modelo trabaje con las descripciones de los textos en el mismo idioma que las descripciones de Audiocaps, se lleva a cabo un gran filtrado de datos, ya que de la base de datos WIT solo es interesante aquellas descripciones que están escritas en inglés, es decir, las que la columna language es ``uk o en". Como consecuencia, se decide complementar la base de datos con nuevos valores utilizando una nueva base de datos Conceptual Captions. Conceptual Captions es una base de datos que busca mostrar de la forma más fiel posible una descripción de las imágenes. Este recurso conceptual de imágenes presenta más de 3 millones de imágenes, junto con subtítulos en lenguaje natural. Estas imágenes y descripciones se recopilan de la Web y representan una variedad muy ampliada de estilos.


Para sacar los embeddings de la base de datos se deben descargar ambas bases de datos: https://github.com/google-research-datasets/wit y https://ai.google.com/research/ConceptualCaptions/ y se siguen los siguientes pasos:


1. Navigate to where the code is:
   ```console
     cd /Audio-to-image-with--AudioLDM.git/Prior steps/WIT
      ```
       
2. En los siguientes scripts solo hay que cambiar el csv con el nombre de la base de datos WIT o Conceptual Captions:
   ```console
     python embedding.py
     ```
3. En este punto se tienen los embeddings de CLIP y CLAP correspondientes en archivos de texto listos para entrenar los modelos translate.























