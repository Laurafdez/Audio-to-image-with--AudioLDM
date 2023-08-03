import pandas as pd
from torchvision import transforms
import requests
from PIL import Image
from io import BytesIO
import torch
import clip
import os
import numpy as np
import laion_clap
from datasets import load_dataset
from datasets.utils.file_utils import get_datasets_user_agent

model_clap = laion_clap.CLAP_Module(enable_fusion=False)
model_clap.load_ckpt()  # Descargar el punto de control preentrenado por defecto.

def get_image_embedding(url, caption):
    if pd.isnull(url):
        print("Invalid URL: Empty or NaN value")
        return None, None
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            imagen_bytes = response.content
            imagen = Image.open(BytesIO(imagen_bytes))
            if imagen is None:
                print("Error al abrir la imagen")
                return None, None
            else:
                imagen = imagen.convert("RGB")  # Convertir la imagen al formato RGB

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                # Cargar el modelo preentrenado de CLIP
                model, preprocess = clip.load("ViT-B/32")
                model = model.to(device)
                model.eval()

                preprocess = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                ])

                imagen = preprocess(imagen).unsqueeze(0).to(device)
                imagen = imagen.div(255.0)  # Normalizar el tensor de imagen al rango [0, 1]

                with torch.no_grad():
                    embed_clip = model.encode_image(imagen)

                text_data = [caption, caption]
                embed_clap = model_clap.get_text_embedding(text_data, use_tensor=True)
                embed_clap = embed_clap[0]

                return embed_clip.detach(), embed_clap
        else:
            print("Error downloading the image:", response.status_code)
            return None, None

    except Image.UnidentifiedImageError:
        print(f"UnidentifiedImageError for URL: {url}")
        return None, None

# Cargar el conjunto de datos "google/wit"
dataset = load_dataset("google/wit", split="train[:5000]")  # Descargar solo los primeros 100 ejemplos de la partición de entrenamiento

# Obtener el subconjunto deseado
split_name = "train"  # Nombre de la partición (por ejemplo, "train", "validation", "test")
subset = dataset[split_name]

# Rutas y configuración del programa
ruta_destino = "/home/byo/Amazon/laura/database/datos_filtrados2/"  # Ruta de destino para los archivos de texto

# Crear la carpeta de destino si no existe
os.makedirs(ruta_destino, exist_ok=True)

count = 2711  # Contador de archivos generados

# Iterar sobre los ejemplos del subconjunto
# Iterar sobre los ejemplos del subconjunto
for example in subset:
    language = example["language"]
    if language not in ["en","uk"]:
        continue  # Saltar al siguiente ejemplo si el idioma no es "en", "o" o "uk"

    url = example["image_url"]
    caption = example["caption"]
    if pd.notnull(caption) and caption.strip():
        print("Processing URL:", url)  # Imprimir la URL en proceso
        print("Processing caption:", caption)
        # Obtener el embedding de la imagen
        embedding_clip, embedding_clap = get_image_embedding(url, caption)
        if embedding_clip is not None and embedding_clap is not None:
            # Guardar los embeddings en un archivo de texto
            ruta_archivo_txt = os.path.join(ruta_destino, f"embedding_{count}.txt")
            with open(ruta_archivo_txt, "w") as file:
                file.write("embedding CLIP\n")
                file.write(str(embedding_clip))
                file.write("\nembedding CLAP\n")
                file.write(str(embedding_clap))
            count += 1
            if count >= 50945:
                break  # Salir del bucle si se han generado suficientes archivos
    else:
        print("Invalid caption: Empty or NaN value")

    if count >= 50945:
        break  # Salir del bucle si se han generado suficientes archivos
