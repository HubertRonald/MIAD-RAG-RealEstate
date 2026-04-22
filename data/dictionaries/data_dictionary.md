# Data Dictionary - real_estate_listings

Este documento describe el mapeo entre el dataset CSV enriquecido y la tabla final en BigQuery para `real_estate_listings`.

## Reglas de normalización aplicadas

- Todos los nombres de columnas en BigQuery se definen en minúsculas.
- Se usa `snake_case`.
- No se usan puntos `.` en nombres de columnas; se reemplazan por `_`.
- Las columnas del CSV se preservan tal como vienen en origen.
- Las columnas de BigQuery reflejan el esquema final de almacenamiento.

## Tabla de mapeo

| columna_csv | columna_bigquery | tipo_sugerido_csv | tipo_sugerido_bigquery | descripcion |
|---|---|---|---|---|
| id | id | object | STRING | Identificador único del anuncio (Primary Key). |
| scraped_at | scraped_at | object | TIMESTAMP | Fecha y hora exacta de la captura del dato (scraping). |
| operation_type | operation_type | object | STRING | Tipo de operación: Venta o Alquiler. |
| property_type | property_type | object | STRING | Categoría del inmueble (Apartamento, Casa, etc.). |
| l3 | l3 | object | STRING | Nivel 3 de localización (barrio, zona específica; en algunos casos incluye dirección). |
| title | title | object | STRING | Título descriptivo de la publicación. |
| description | description | object | STRING | Descripción detallada del inmueble. |
| status | status | object | STRING | Estado de la publicación en la plataforma. |
| seller_name | seller_name | object | STRING | Nombre del anunciante o inmobiliaria. |
| seller_type | seller_type | object | STRING | Tipo de vendedor (dueño directo o inmobiliaria). |
| seller_id | seller_id | float64 | FLOAT | Identificador del vendedor. |
| url | url | object | STRING | Enlace directo a la publicación original. |
| image_urls | image_urls | object | STRING | Lista de enlaces a las fotografías del inmueble. |
| thumbnail_url | thumbnail_url | object | STRING | Enlace a la imagen miniatura de la publicación. |
| bedrooms | bedrooms | float64 | FLOAT | Cantidad de dormitorios (validados o imputados). |
| bathrooms | bathrooms | float64 | FLOAT | Cantidad de baños (validados o imputados). |
| garages | garages | float64 | FLOAT | Cantidad de espacios de estacionamiento disponibles. |
| surface_total | surface_total | float64 | FLOAT | Superficie total del inmueble en m². |
| surface_covered | surface_covered | float64 | FLOAT | Superficie construida en m². |
| floor | floor | float64 | FLOAT | Número de piso (específico para apartamentos). |
| age | age | float64 | FLOAT | Antigüedad del inmueble en años o año de construcción. |
| condition | condition | object | STRING | Estado de la propiedad (Nuevo/Usado). |
| expenses | expenses | float64 | FLOAT | Monto de gastos comunes o expensas. |
| amenities | amenities | object | STRING | Lista de servicios adicionales (Piscina, Gym, Barbacoa, etc.). |
| price | price | float64 | FLOAT | Precio original publicado. |
| currency | currency | object | STRING | Moneda original de la publicación (USD, UYU). |
| price_fixed | price_fixed | float64 | FLOAT | Campo ETL. Precio normalizado (USD para ventas, UYU para alquileres). |
| currency_fixed | currency_fixed | object | STRING | Campo ETL. Moneda de referencia tras la normalización. |
| price_m2 | price_m2 | float64 | FLOAT | Campo ETL. Precio calculado por metro cuadrado. |
| price_m2_basis | price_m2_basis | object | STRING | Campo ETL. Indica si el cálculo m² se basó en área total o cubierta. |
| lat | lat | float64 | FLOAT | Latitud geográfica del inmueble. |
| lon | lon | float64 | FLOAT | Longitud geográfica del inmueble. |
| geometry | geometry | object | STRING | Objeto geométrico Point para procesamiento espacial. |
| BARRIO | barrio | object | STRING | Nombre del barrio oficial de Montevideo (validado por Join Espacial). |
| NROBARRIO | nrobarrio | float64 | FLOAT | Código numérico oficial del barrio. |
| DEPARTAMEN | departamen | object | STRING | Departamento administrativo (Montevideo). |
| ZONA_LEGAL | zona_legal | object | STRING | Jurisdicción legal (CCZ) asociada a la ubicación. |
| SECCION_POL | seccion_pol | float64 | FLOAT | Sección policial correspondiente. |
| CODBA | codba | object | STRING | Código de base asociado al barrio. |
| n_public_spaces_800m | n_public_spaces_800m | int64 | INTEGER | Cantidad de espacios públicos en un radio de 800m. |
| dist_nearest_public_space | dist_nearest_public_space | float64 | FLOAT | Distancia al espacio público más cercano. |
| public_space_area_800m | public_space_area_800m | float64 | FLOAT | Área total de espacios públicos en un radio de 800m. |
| n_espacio_libre_800m | n_espacio_libre_800m | int64 | INTEGER | Cantidad de espacios libres en 800m. |
| dist_espacio_libre | dist_espacio_libre | float64 | FLOAT | Distancia al espacio libre más cercano. |
| area_espacio_libre_800m | area_espacio_libre_800m | float64 | FLOAT | Área de espacios libres en 800m. |
| n_plaza_800m | n_plaza_800m | int64 | INTEGER | Cantidad de plazas en 800m. |
| dist_plaza | dist_plaza | float64 | FLOAT | Distancia a la plaza más cercana. |
| area_plaza_800m | area_plaza_800m | float64 | FLOAT | Área de plazas en 800m. |
| n_ord.transito_800m | n_ord_transito_800m | int64 | INTEGER | Cantidad de áreas de ordenamiento de tránsito en 800m. |
| dist_ord.transito | dist_ord_transito | float64 | FLOAT | Distancia al área de tránsito más cercana. |
| area_ord.transito_800m | area_ord_transito_800m | float64 | FLOAT | Área de ordenamiento de tránsito en 800m. |
| n_plazoleta_800m | n_plazoleta_800m | int64 | INTEGER | Cantidad de plazoletas en 800m. |
| dist_plazoleta | dist_plazoleta | float64 | FLOAT | Distancia a la plazoleta más cercana. |
| area_plazoleta_800m | area_plazoleta_800m | float64 | FLOAT | Área de plazoletas en 800m. |
| n_isla_800m | n_isla_800m | int64 | INTEGER | Cantidad de islas en 800m. |
| dist_isla | dist_isla | float64 | FLOAT | Distancia a la isla más cercana. |
| area_isla_800m | area_isla_800m | float64 | FLOAT | Área de islas en 800m. |
| n_playa_800m | n_playa_800m | int64 | INTEGER | Cantidad de playas en 800m. |
| dist_playa | dist_playa | float64 | FLOAT | Distancia a la playa más cercana. |
| area_playa_800m | area_playa_800m | float64 | FLOAT | Área de playas en 800m. |
| n_escuelas_800m | n_escuelas_800m | int64 | INTEGER | Cantidad de escuelas (ANEP) en 800m. |
| dist_nearest_escuela | dist_nearest_escuela | float64 | FLOAT | Distancia a la escuela más cercana. |
| n_primaria_800m | n_primaria_800m | int64 | INTEGER | Cantidad de centros de primaria en 800m. |
| dist_primaria | dist_primaria | float64 | FLOAT | Distancia al centro de primaria más cercano. |
| n_secundaria_800m | n_secundaria_800m | int64 | INTEGER | Cantidad de centros de secundaria en 800m. |
| dist_secundaria | dist_secundaria | float64 | FLOAT | Distancia al centro de secundaria más cercano. |
| n_tecnica_800m | n_tecnica_800m | int64 | INTEGER | Cantidad de centros de educación técnica en 800m. |
| dist_tecnica | dist_tecnica | float64 | FLOAT | Distancia al centro de educación técnica más cercano. |
| n_formacion_docente_800m | n_formacion_docente_800m | int64 | INTEGER | Cantidad de centros de formación docente en 800m. |
| dist_formacion_docente | dist_formacion_docente | float64 | FLOAT | Distancia al centro de formación docente más cercano. |
| n_destinos_800m | n_destinos_800m | int64 | INTEGER | Cantidad de puntos de destino en 800m. |
| dist_nearest_destino | dist_nearest_destino | float64 | FLOAT | Distancia al punto de destino más cercano. |
| n_comercial_800m | n_comercial_800m | int64 | INTEGER | Cantidad de destinos comerciales en 800m. |
| dist_comercial | dist_comercial | float64 | FLOAT | Distancia al destino comercial más cercano. |
| n_gubernamental_800m | n_gubernamental_800m | int64 | INTEGER | Cantidad de destinos gubernamentales en 800m. |
| dist_gubernamental | dist_gubernamental | float64 | FLOAT | Distancia al destino gubernamental más cercano. |
| n_industrial_800m | n_industrial_800m | int64 | INTEGER | Cantidad de destinos industriales en 800m. |
| dist_industrial | dist_industrial | float64 | FLOAT | Distancia al destino industrial más cercano. |

## Notas de implementación

- `real_estate_listings` es la fuente de verdad estructurada del sistema.
- El índice FAISS no reemplaza esta tabla; almacena embeddings y referencias por `property_id`.
- Las columnas con nombres originales en mayúsculas o con punto se normalizan para BigQuery.
- Si el CSV conserva listas u objetos serializados, campos como `image_urls`, `amenities` y `geometry` pueden cargarse inicialmente como `STRING`.