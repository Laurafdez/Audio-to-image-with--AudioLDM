# Audio-to-image
Este repositorio va a mostrar un modelo que utiliza AudioLDM para sonorizar imagenes. Para ello, se ha utilizado el modelo AudioLDM preentrenado para poder generar audios cuando recibe como entrada una imagen. Se han seguido distintos paso para llegar a dicha implementación. AudioLDM es un modelo que es capaz de generar audio a partir de un texto, para poder realizar el proceso cuenta en su implementación con el modelo CLAP. La función principal de CLAP es la de condicionar los audios generados en función de un texto a la entrada o lo que es lo mismo, los embeddings que salen del módulo CLAP son vectores que condicionan la generación de audio por el texto. AudioLDM utiliza el codificador de CLAP para codificicar sus textos. Este hecho se va a aprovechar para desarrollar una red neuronal Traductor que sea capaz de traducir los embeddings de CLIP en embeddings que se parezcan a embeddings de CLAP, controlando este proceso con la ayuda de una función de coste. Se busca traducir los embeddings de CLIP en embeddings de CLAP, los embeddings de las imágenes que codifica CLIP están condicionados por textos y los embeddings de los textos por las imágenes. Como el objetivo de este modelo es generar audio a partir de una imagen resulta conveniente usar CLIP para este proceso. CLIP es modelo multimodal creado por la empresa OpenAI que utiliza el aprendizaje zero shot y utiliza NLP para supervisar el entrenamiento, es decir, es capaz de trabajar con un dato que no ha visto antes solo conociendo su descripción o nombre. CLIP está formado por una red neuronal que ha sido entrenado por una variedad de pares(imagen, texto). Busca que dada una imagen es capaz de encontrar el mejor texto para esa imagen. Una vez que se tenga el modelo Traductor creado, se sustituirá el encoder de texto de AudioLDM por este modelo para conseguir que AudioLDM sea capaz de generar sonido dada una imagen.

En este repositprio se van a explicar los pasos seguidos para desarrollar este proceso.

## Fase I. Base de datos

El primer paso es sacar los embeddings de las distintas bases de datos que se han utlizado para el entrenamiento de los modelos translate. Primero de todo, se comienza a sacar los embeddings de la base de datos que utilizó AudioLDM para su entrenamiento, esta fue Audiocaps. Audiocaps es una base de datos que se encarga de describir en lenguaje natural cualquier tipo de audio en condiciones reales. Esta base de datos consta de 57000 pares de clips de audios y descripciones de texto escrito por humanos.

La columna que es importante para la implementación de la red neuronal es la de captions, ya que en ella se encuentran las descripciones de los sonidos. Una vez descargada toda la base de datos se procede a obtener los embeddings de la columna captions tanto por el codificador de CLIP como por el codificador de CLAP, ya que lo que se busca es desarrollar un modelo que sea capaz de traducir unos embeddings en otros.

Se realiza el proceso de codificación de cada una de las captions por medio del codificador de texto de CLIP y de CLAP. Se genera un archivo de texto en el que se recogen los embeddings de CLIP y de CLAP para una misma descripción de textos. Estos embeddings son vectores de 1x512. Es importante conocer si los valores se encuentran normalizados o no, por ello, se sacan los histogramas de las normas de ambos embeddings.

En la siguiente imagen se puede observar como los histogramas de CLIP no se encuentran normalizados, este es un dato muy importante porque a la hora de diseñar la red neuronal a de tenerse en cuenta, ya que la entrada del módulo deberá normalizar los valores antes de trabajar con ellos.



Por otro lado, para los embeddings de CLAP, observables en la siguiente imagen, sus normas sí que están normalizadas entre 0 y 1, es decir, en este caso no es necesario que a la salida de la red neuronal se haga un proceso de desnormalización.


Para poder reproducir el proceso de sacar los embeddings de las base de datos y sacar sus espectrogramas estos son los pasos que hay que seguir:




























