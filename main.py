from pathlib import Path
import pandas as pd
from modules import normal as n

ruta = Path("data")

# 1. Buscar archivos .csv
archivos_csv = [p for p in ruta.iterdir() if p.is_file() and p.suffix == ".csv"]

if not archivos_csv:
    print("No hay archivos CSV en la carpeta data/")
    exit()

# 2. Mostrar lista numerada
print("Archivos CSV encontrados:")
for i, archivo in enumerate(archivos_csv, start=1):
    print(f"{i}. {archivo.name}")

# 3. Pedir número al usuario
opcion = int(input("\nElige un archivo escribiendo su número: "))

# Validar opción
if opcion < 1 or opcion > len(archivos_csv):
    print("Opción inválida.")
    exit()

# 4. Seleccionar archivo
archivo_seleccionado = archivos_csv[opcion - 1]
archivo_nombre = str(archivo_seleccionado)[5:-4]
df = pd.read_csv(archivo_seleccionado)

print("\nLet me do my Magic!")
df, encoders = n.procesar_categoricas(df)
normalized_df, scaler = n.normalize_dataset(df, archivo_nombre)

print(df.head)

