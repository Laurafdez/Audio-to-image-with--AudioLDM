import numpy as np
import laion_clap
import pandas as pd
import numpy as np
import pandas as pd



# Leer el archivo CSV
df = pd.read_csv("train.csv")

df = df.drop(["audiocap_id", "youtube_id", "start_time"], axis=1)




# quantization
def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)


def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)


model = laion_clap.CLAP_Module(enable_fusion=False)
model.load_ckpt()  # download the default pretrained checkpoint.

# Crear un nuevo DataFrame para almacenar los resultados
df_embed = pd.DataFrame(columns=["caption", "embedding", "tensor"])

# Iterar sobre los textos de las columnas "caption" en lotes de 10
for i in range(0, len(df), 10):
    # Obtener los textos de las columnas "caption" en el lote actual
    batch_captions = df["caption"].iloc[i:i + 10].tolist()

    # Obtener los embeddings y tensores para el lote actual
    batch_embed = model.get_text_embedding(batch_captions, use_tensor=True)

    # Crear un DataFrame temporal para el lote actual
    df_batch = pd.DataFrame()
    df_batch["caption"] = batch_captions
    df_batch["embedding"] = batch_embed.tolist()

    # Concatenar el DataFrame del lote actual al DataFrame principal
    df_embed = pd.concat([df_embed, df_batch], ignore_index=True)

folder_path = "C:/Users/laura/audio/"
# Guardar el DataFrame con los embeddings en un archivo CSV
df_embed.to_csv(folder_path + "CLAP.csv", index=False)