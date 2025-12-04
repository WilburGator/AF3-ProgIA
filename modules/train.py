from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import pandas as pd

def entrenar_logistic_regression(df, nombre_dataset):
    print("\nColumnas disponibles:")
    for col in df.columns:
        print(f" - {col}")

    target = input("\nEscribe el nombre del objetivo: ").strip()

    if target not in df.columns:
        raise ValueError(f"La columna '{target}' no existe en el dataframe.")

    X = df.drop(columns=[target])
    y = df[target]

    carpeta_modelos = Path("models")
    carpeta_modelos.mkdir(exist_ok=True)

    base = nombre_dataset[:-4]
    ruta_modelo = carpeta_modelos / f"{base}_logreg.pkl"

    if ruta_modelo.exists():
        print("Modelo ya existente. Cargando modelo entrenado.")
        modelo = joblib.load(ruta_modelo)
        return modelo

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print("\nEntrenando modelo Logistic Regression...")
    modelo = LogisticRegression(max_iter=200)
    modelo.fit(X_train, y_train)

    preds = modelo.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Precisión en test: {acc:.4f}")

    joblib.dump(modelo, ruta_modelo)
    print(f"Modelo guardado en: {ruta_modelo}")

    return modelo

def entrenar_logistic_regression_multinomial(df, nombre_dataset):
    print("\nColumnas disponibles:")
    for col in df.columns:
        print(f"- {col}")

    target = input("\nEscribe el nombre de la columna objetivo: ").strip()

    if target not in df.columns:
        raise ValueError(f"La columna '{target}' no existe en el dataframe.")


    X = df.drop(columns=[target])
    y = df[target]

    carpeta_modelos = Path("models")
    carpeta_modelos.mkdir(exist_ok=True)

    base = nombre_dataset[:-4]
    ruta_modelo = carpeta_modelos / f"{base}_logreg_multinomial.pkl"

    if ruta_modelo.exists():
        print("✔ Modelo multinomial ya existente. Cargando modelo entrenado.")
        modelo = joblib.load(ruta_modelo)
        return modelo

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=32
    )

    print("\nEntrenando modelo Logistic Regression (multinomial).")
    modelo = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=300
    )
    modelo.fit(X_train, y_train)

    preds = modelo.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"✔ Precisión (multinomial) en test: {acc:.4f}")

    joblib.dump(modelo, ruta_modelo)
    print(f"✔ Modelo multinomial guardado en: {ruta_modelo}")

    return modelo