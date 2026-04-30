import pandas as pd
import base64
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(BASE_DIR, "valid_barrios__epsg4326_210426.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "../samples/real_estate_listings.csv")

# columnas que NO se deben anonimizar (aunque sean string)
EXCLUDE_COLUMNS = [
    "scraped_at",
    "geometry",
    "operation_type",
    "property_type",
    "currency",
    "currency_fixed",
    "price_m2_basis",
    "barrio",  # clave para RAG
]

def encode_base64(value):
    if pd.isna(value):
        return value
    try:
        return base64.b64encode(str(value).encode("utf-8")).decode("utf-8")
    except Exception:
        return value

def main():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"No se encontro el archivo: {INPUT_FILE}")

    print("Leyendo CSV...")
    df = pd.read_csv(INPUT_FILE)

    print(f"Total registros: {len(df)}")

    print("Tomando muestra reproducible...")
    df_sample = df.sample(n=10, random_state=42)

    print("Detectando columnas tipo string...")
    string_columns = df_sample.select_dtypes(include=["object"]).columns.tolist()

    print("Columnas string detectadas:")
    print(string_columns)

    print("Aplicando anonimización selectiva...")
    for col in string_columns:
        if col not in EXCLUDE_COLUMNS:
            df_sample[col] = df_sample[col].apply(encode_base64)

    print("Creando carpeta samples...")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    print("Guardando CSV anonimizado...")
    df_sample.to_csv(OUTPUT_FILE, index=False)

    print(f"Sample generado en: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()