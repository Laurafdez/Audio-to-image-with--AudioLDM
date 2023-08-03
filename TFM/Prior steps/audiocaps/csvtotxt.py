import pandas as pd
import numpy as np
import os
import csv

clip_embeddings_file='processed_embeddings2.csv'
clap_embeddings_file='CLAP.csv'
directorio ='C:/Users/laura/audiocaps/dataset/buenos/clip'

with open(clip_embeddings_file, 'r') as clip_file, open(clap_embeddings_file, 'r') as clap_file:
    clip_reader = csv.reader(clip_file)
    clap_reader = csv.reader(clap_file)

    # Omitir la primera fila de cada archivo (encabezado)
    next(clip_reader)
    next(clap_reader)

    for i, (clip_row, clap_row) in enumerate(zip(clip_reader, clap_reader)):
        nombre_archivo = f'muestra_{i}.txt'
        ruta_archivo = f'{directorio}/{nombre_archivo}'

        with open(ruta_archivo, 'w') as archivo:
                archivo.write('CLIP Embedding:\n')
                np.savetxt(archivo, np.array(clip_row).astype(np.float64), delimiter=',')
                
                archivo.write('\n\nCLAP Embedding:\n')
                clap_row_values = ' '.join(clap_row).strip('[]').split(', ')
                clap_row_float = [float(value) for value in clap_row_values]
                np.savetxt(archivo, np.array(clap_row_float), delimiter=',')

