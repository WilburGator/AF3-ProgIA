from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
import pandas as pd
import joblib
import os
from pathlib import Path

def normalize_dataset(df, dataset_name):
    ver_ruta = "normalized_data/" + dataset_name
    print(ver_ruta)

    if Path(ver_ruta).exists():
        print("Este dataset ya fue normalizado.")
        # cargar scaler
        scaler = joblib.load(ver_ruta + "/scaler.pkl")

        # cargar data normalizada
        df_norm = pd.read_csv(ver_ruta + "/dataset.csv")

        print("Datos normalizados y scaler cargados correctamente.")
        return df_norm, scaler

    # Crear carpetas si no existen
    os.makedirs("normalized_data", exist_ok=True)
    os.makedirs(ver_ruta, exist_ok=True)

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found to normalize.")

    # 2. Crear y ajustar el scaler
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # 3. Guardar el scaler con el nombre del dataset
    scaler_filename = f"normalized_data/{dataset_name}/scaler.pkl"
    joblib.dump(scaler, scaler_filename)

    # 4. Guardar el dataset normalizado
    normalized_filename = f"normalized_data/{dataset_name}/dataset.csv"
    df.to_csv(normalized_filename, index=False)

    print(f"Scaler guardado en: {scaler_filename}")
    print(f"Dataset normalizado guardado en: {normalized_filename}")

    return df, scaler

def OHE(df, col):
    try:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop=None)
    except TypeError:
        ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')

    transform = ohe.fit_transform(df[[col]])
    n_cols = ohe.get_feature_names_out([col])
    df_ohe = pd.DataFrame(transform, columns=n_cols, index=df.index)
    df = pd.concat([df.drop(columns=[col]), df_ohe], axis=1)
    return df, ohe

def le_encode(df, col):
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    return df, le

def procesar_categoricas(df):
    df = df.copy()
    
    # Detectar columnas categóricas
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if not cat_cols:
        print("No se encontraron columnas categóricas.")
        return df, {}
    
    encoders = {}
    
    print("Columnas categóricas detectadas:")
    for col in cat_cols:
        print(f" - {col}")
    
    # Procesar columnas una por una
    for col in cat_cols:
        print(f"\n¿Cómo deseas codificar la columna '{col}'?")
        print("0. Ignorar")
        print("1. OneHotEncoder (OHE)")
        print("2. LabelEncoder")
        opcion = input("Elige 0, 1 o 2: ")
        
        while opcion not in ["1", "2", "0"]:
            opcion = input("Opción inválida. Elige 1 o 2: ")
        
        if opcion == "1":
            df, ohe = OHE(df, col)
            encoders[col] = ("OHE", ohe)
        
        if opcion == "2":
            df, le = le_encode(df, col)
            encoders[col] = ("LE", le)
    
    print("\nProceso completado.")
    return df, encoders