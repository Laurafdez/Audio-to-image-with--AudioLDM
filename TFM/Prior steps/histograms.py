import csv
import matplotlib.pyplot as plt
import numpy as np

ruta_csv = 'CLIP.csv'  # Ruta de tu archivo CSV

normas = []

with open(ruta_csv, 'r') as archivo:
    lector_csv = csv.reader(archivo)
    next(lector_csv)  

    for fila in lector_csv:
        valores = [float(valor) for valor in fila]  # Convertir los valores a tipo numérico si es necesario
        norma = np.linalg.norm(valores)
        normas.append(norma)

tamano = len(normas)

for norma in normas:
    print("Valor de la norma:", norma)

print("Tamaño del array normas:", tamano)
# Suponiendo que deseas mostrar el valor de la primera norma
valor_norma = normas[0]
print("Valor de la norma:", valor_norma)


plt.hist(normas, bins=10)  # Crear el histograma con 10 bins (ajústalo según tus necesidades)
plt.xlabel('Norma')
plt.ylabel('Frecuencia')
plt.title('Histograma de las normas de las filas')
plt.show()
