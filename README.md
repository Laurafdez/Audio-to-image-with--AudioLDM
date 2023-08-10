# Audio-to-image
This repository is going to show a model that uses AudioLDM to sound images. For this, the AudioLDM model has been pre-trained to be able to generate audio when it receives an image as input. Several steps have been followed to reach this implementation. AudioLDM is a model that is able to generate audio from a text, to be able to carry out the process it counts on the CLAP model in its implementation. The main function of CLAP is to condition the audio generated based on a text input or, in other words, the embeddings that come out of the CLAP module are vectors that condition the audio generation based on the text. AudioLDM uses the CLAP encoder to encode its texts. This fact will be used to develop a Translator neural network capable of translating CLIP embeddings into embeddings that resemble CLAP embeddings, controlling this process with the help of a cost function. The aim is to translate CLIP embeddings into CLAP embeddings, the embeddings of the images encoded by CLIP are conditioned by texts and the embeddings of the texts by the images. As the objective of this model is to generate audio from an image, it is convenient to use CLIP for this process. CLIP is a multimodal model created by the company OpenAI that uses zero-shot learning and uses NLP to supervise the training, i.e. it is able to work with a piece of data that it has not seen before just by knowing its description or name. CLIP consists of a neural network that has been trained on a variety of pairs (image, text). It finds that given an image it is able to find the best text for that image. Once the Translator model has been created, the AudioLDM text encoder will be replaced by this model to make AudioLDM able to generate sound given an image.

<div align="center">
  <img src="TFM/Arquitectura AudioLDM.png" />
</div>


## Data bases

The first step is to extract the embeddings from the different databases that have been used to train the translate models. First of all, we start by extracting the embeddings from the database that AudioLDM used for its training, which was Audiocaps. Audiocaps is a database that is in charge of describing in natural language any kind of audio in real conditions. This database consists of 57000 pairs of audio clips and text descriptions written by humans.

The column that is important for the implementation of the neural network is the captions column, as it contains the descriptions of the sounds. Once the entire database has been downloaded, the embeddings of the captions column are obtained by both the CLIP encoder and the CLAP encoder, since the aim is to develop a model that is capable of translating some embeddings into others.

The encoding process of each of the captions is carried out by means of the CLIP and CLAP text encoder. A text file is generated in which the CLIP and CLAP embeddings for the same text description are collected. These embeddings are 1x512 vectors. It is important to know whether the values are normalised or not, so the histograms of the norms of both embeddings are extracted.

In the following image it can be seen how the CLIP histograms are not normalised, this is a very important fact because it must be taken into account when designing the neural network, since the module input must normalise the values before working with them.

<div align="center">
  <img src="TFM/CLIPhis.png" width="300" height="250"  />
</div>

On the other hand, for the CLAP embeddings, which can be seen in the following image, their norms are normalised between 0 and 1, i.e., in this case it is not necessary to perform a denormalisation process at the output of the neural network.

<div align="center">
  <img src="TFM/CLAPhis.png" width="300" height="250" />
</div>

In order to reproduce the process of extracting the embeddings from the Audiocaps database, it is necessary to download it from the following link https://audiocaps.github.io/ y sacar los siguientes pasos:


1. Clone the repository and navigate to where the code is:
   ```console
      git clone https://github.com/Laurafdez/Audio-to-image-with--AudioLDM.git
      ```
2. Navigate to where the code is:
   ```console
     cd /Audio-to-image-with--AudioLDM.git/TFM/Prior steps/audiocaps
      ```
       
2. In the following scripts just change the csv with the name of the Audiocaps database and execute:
   ```console
     python embedding_CLIP.py
     ```
3. And to get the CLAP embeddings:
   ```console
     python embedding_CLAP.py
     ```
4. Two csv's will have been created with the CLIP and CLAP embeddings, to create text files with this information run the following script:
   ```console
     python csvtotxt.py
     ```
5.  At this point you will have all the CLIP and CLAP embeddings in text files for each of the Adiocaps data.



With these data, 6 models of Translate will be trained with different cost and intermediate layer functions: 256 or 512. And the cost functions are: cosine distance, mean square error and constractive learning.

In addition, as the aim is to generate audios that are as faithful as possible to the images, it was decided to train 12 more models, the difference between these models is that 6 models will be trained first, whose database is the same as the one used by CLIP in its training. In this way the new Translator models will work with the embeddings directly from the images and will translate them into embeddings similar to the CLAP embeddings. 

In order to carry out this process, two new databases have been downloaded. On the one hand, the database that CLIP uses for training, WIT: Wikipedia-based image text dataset. WIT is a Wikipedia-based image text dataset, a multilingual multimodal dataset consisting of 37.6 million text examples with 11.5 million unique images in 108 languages. This dataset was created by extracting multiple texts associated with an image from Wikipedia articles and links from Wikimedia images. It is the largest dataset in existence to date. It is mainly used for pre-training of multimodal models and features high quality alignment between images and texts. 

A script is implemented to download each of these images and extract the corresponding embeddings using the CLIP encoder. Likewise, the corresponding descriptions of these images are encoded with the help of the CLAP encoder. Each of these embeddings, both CLIP and CLAP, are saved in txt files. To complement the downloaded data, it was decided to supplement the database with new values using a new Conceptual Captions database. Conceptual Captions is an image database of more than 3 million images, together with natural language captions. These images and descriptions are collected from the Web and represent a very wide variety of styles.


To extract the embeddings from the database, download both databases: https://github.com/google-research-datasets/wit and https://ai.google.com/research/ConceptualCaptions/ and follow the steps below:


1. Navigate to where the code is:
   ```console
     cd /Audio-to-image-with--AudioLDM.git/TFM/Prior steps/WIT
      ```
       
2. In the following scripts just change the csv with the name of the WIT or Conceptual Captions database:
   ```console
     python embedding.py
     ```
3. At this point you have the corresponding CLIP and CLAP embeddings in text files ready to train the translate models.

## Translate

Once the database of images and texts describing the sounds has been built, we proceed to train the models, in total 18 models are built for the different databases, inside the TFM folder we find the files Models Audiocaps, models WIT and models with Audiocaps and WIT, with the scripts to implement the different models with the different databases. Of each type, 6 models are trained, the only data to be changed is the dimension of the hidden layer from 256 to 512. The shape of the neural network is as follows:

<div align="center">
  <img src="TFM/translator.png" width="460" height="350" />
</div>


Each of the models will be trained and it will take a while to give an example of how it should be done for one of the models:

1. Navigate to where the code is:
   ```console
     cd /Audio-to-image-with--AudioLDM.git/TFM/Models_Audiocaps
      ```
       
2. The database is trained for the model whose database is Audiocaps, its hidden layer dimension of value 256 and its cost function is the mean square error:
   ```console
     python traductorEMC.py
     ```
3. To train the model with a 512 heat dimension layer, just change that data in the code.
4. The same is done to train the model whose cost function is the cosine distance:
   ```console
     python traductorcos.py
     ```
5. And the previous script will be run again, changing the data of the hidden layer to 512.
6. The same two steps above to train the model whose cost function is Constractive Learning and run:
   ```console
     python traductorCL.py
     ```
7.  At this point 6 models trained with the Audiocaps database are obtained. To train the rest of the models with the different databases, all the previous steps must be followed, but entering and executing the scripts found in Models_WIT+Audiocaps and Models_WIT. By executing each of the files and changing the hidden dimension layer from 256 to 512 in each of them, a total of 12 more models are obtained.

## Implementation within AudioLDM

Once the trained models were obtained, we proceeded to introduce the architecture created within the AudioLDM model. First of all, the code was downloaded from the AudioLDM repository https://github.com/haoheliu/AudioLDM, to which all the modifications were made so that it could receive an image as input. As we wanted the model to be able to work when receiving a text input or an image input. In order to be able to do this process, the methods are implemented following the same methodology for when it receives a text or an image. Now the function is allowed to receive the url of the image so that the model can download it from the Internet and carry out the audio generation process.

Once the code is allowed to receive a link with the photo instead of the image description, the code is adapted to work with that case. All the relevant changes are made so that the new implementation can be carried out. The most important change occurs when the code calls the CLAP module to perform the text encoding, now instead of calling that module it calls a new implemented class.

In this new implemented class is where the flow goes when the code was pulling the embedding from the text, now it will go to this class where the translate model will do the translation process. First, the image will be downloaded from the url provided, then the embedding will be extracted using the CLIP model. The next step is to get the CLAP-like embeddings using the translate model. Once the embeddings are obtained, the vector is returned to the stream and AudioLDM continues its process until audio is generated with the new embedding. Finally, audio is generated through an image as input by making these changes. The fact that the value of the errors is so small makes sense given that, as they are normalised, when squared, the numbers will become smaller and smaller.

In order to carry out this process:

1. Navigate to where the code is:
   ```console
     cd /Audio-to-image-with--AudioLDM.git/AudioLDMv1/audioldm
      ``` 
2. In order to generate audio with one of the trained models it is necessary to go to the file image_to_audio and write the name of the model you want to generate audio and write the size of the layers of the model.

3. Once this modification has been made, the script is executed:
   ```console
     python audioldm -t [URL imagen]
      ``` 
4. After a few minutes in the output folder, the audio for this image will be created.






















