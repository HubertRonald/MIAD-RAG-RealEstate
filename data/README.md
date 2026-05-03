# ETL y Análisis de Datos Inmobiliarios - Montevideo 
## Proyecto "De Listado a Asesor Inteligente: Analítica y Recomendación para el Mercado Inmobiliario de Montevido, Uruguay 🇺🇾" 

Este notebook realiza un proceso de ETL avanzado de datos de inmuebles en Montevideo, incluyendo limpieza, detección de anomalías (outliers) y procesamiento de datos geográficos para el posterior uso de estos datos en un modelo RAGAS de recomendación de propiedades. 

## 📁 Estructura del Repositorio

Para garantizar la correcta ejecución del código, el repositorio debe mantener la siguiente estructura:

*   **/inputs**: Contiene el notebook y todos los archivos requeridos para ejecutarlo.
*   **/outputs**: Destino donde serán guardados los archivos generados por el notebook
*   `requirements.txt`: Lista de dependencias y versiones necesarias.

---

## 💻 Requisitos de Entorno

El proyecto fue desarrollado y testeado bajo las siguientes especificaciones técnicas:

*   **Lenguaje:** Python 3.11.5
*   **Sistema Operativo:** macOS (Arquitectura arm64)
*   **Interfaz:** IPython 9.13.0 / Jupyter Client 8.8.0

---

## 🚀 Instrucciones de Configuración y Uso

### 1. Preparación del Entorno
Se recomienda el uso de un entorno virtual para evitar conflictos de versiones. Para instalar las dependencias necesarias, ejecute:
```bash
pip install -r requirements.txt 
```

## 📊 Resultados del Proceso de ETL (Outputs)

Al finalizar la ejecución del proceso, el notebook genera dos archivos CSV en la carpeta correspondiente. Estos representan el dataset validado con diferentes sistemas de referencia de coordenadas (CRS):

**`valid_barrios_proj_290426.csv`**:
    *   **Unidad:** Metros.
    *   **Descripción:** Datos proyectados para cálculos métricos precisos. Es el archivo indicado para análisis de superficies y distancias lineales dentro del departamento de Montevideo.

**`real_estate_listings.csv`**:
    *   **Unidad:** 
    Grados decimales.
    *   **Descripción:** 
    Utiliza el estándar WGS84 (EPSG:4326). Es el formato utilizado por el modelo RAGAS ya que es necesario para integrar los datos con herramientas de mapas web (como Folium o Leaflet) y visualizaciones basadas en Latitud/Longitud.

