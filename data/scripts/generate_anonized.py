import pandas as pd
import base64
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(BASE_DIR, "../inputs/raw_outputMLScrapper250226.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "../inputs/outputMLScrapper250226.csv")

# columnas que NO se deben anonimizar (aunque sean string)
EXCLUDE_COLUMNS = (
    "operation_type,property_type,price,currency,"
    "status,condition,neighborhood,city,state,latitude,longitude,"
    "surface_total,surface_covered,bedrooms,bathrooms,garages,floor,"
    "age,expenses,amenities,seller_name,seller_type,seller_id,"
    "description,scraped_at"
)

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
    df_sample = df.sample(n=30, random_state=42)

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