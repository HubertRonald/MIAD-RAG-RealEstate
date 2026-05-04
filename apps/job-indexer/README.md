# Resumen

Se realizaron pruebas de carga controladas del `job-indexer` para validar la generación del índice FAISS a partir de BigQuery y embeddings de Gemini. La primera corrida completa con `cpu=1`, `memory=1Gi` y `BQ_LIMIT=null` no fue viable: después de aproximadamente 17 minutos la instancia fue terminada por falta de memoria. Posteriormente, se probaron cargas limitadas: con `2 CPU / 2Gi` y `BQ_LIMIT=100` el job finalizó en 1.83 minutos; con `2 CPU / 4Gi` y `BQ_LIMIT=500` finalizó en 4 minutos; con `4 CPU / 4Gi` y `BQ_LIMIT=1000` finalizó en 6.9 minutos, mostrando una mejora moderada pero no lineal. Finalmente, la indexación completa de cerca de 3500 registros con `4 CPU / 8Gi`, `timeout=3600s`, `max_retries=0` y `BQ_LIMIT=null` finalizó exitosamente en 26.18 minutos. En conjunto, las pruebas muestran que el cuello de botella no depende únicamente de CPU, sino también de memoria, latencia de embeddings, batching, serialización y construcción local del índice FAISS.


# Pruebas de carga, limitaciones e industrialización del Job Indexer

## Pruebas de carga controladas

Durante la migración del pipeline local hacia `apps/job-indexer`, se realizaron varias pruebas de carga para validar la capacidad del Cloud Run Job encargado de construir el índice FAISS desde BigQuery y publicarlo en Cloud Storage.

| Configuración | BQ_LIMIT | Resultado           |  Tiempo aproximado | Registros/min aprox. |
| ------------- | -------: | ------------------- | -----------------: | -------------------: |
| 1 CPU / 1Gi   |     null | Falló por memoria   | 17 min hasta caída |                  N/A |
| 2 CPU / 2Gi   |      100 | Éxito               |           1.83 min |                 54.6 |
| 2 CPU / 4Gi   |      500 | Éxito               |              4 min |                  125 |
| 4 CPU / 4Gi   |     1000 | Éxito               |            6.9 min |                  145 |
| 4 CPU / 8Gi   |     null | Éxito full indexing |          26.18 min |                 ~134 |

La prueba inicial con `1 CPU / 1Gi` y corrida completa no fue suficiente, ya que el contenedor fue terminado por falta de memoria. Esto permitió confirmar que el proceso no solo depende del número de registros, sino también de la memoria requerida para mantener en ejecución el DataFrame, los documentos LangChain, los textos procesados, los embeddings, los artefactos temporales, el vectorstore FAISS y los archivos auxiliares.

La configuración final usada para indexación completa fue:

```hcl
cpu              = "4"
memory           = "8Gi"
timeout          = "3600s"
max_retries      = 0
indexer_bq_limit = null
```

El uso de `max_retries = 0` fue intencional para evitar que una falla por memoria, permisos o cuota generara reintentos automáticos y, por tanto, posibles llamadas repetidas a Gemini Embeddings.

## Limitaciones actuales del prototipo

La implementación actual funciona correctamente para una reconstrucción completa del índice, pero todavía responde a una lógica de prototipo batch. El job lee la tabla completa o una muestra limitada desde BigQuery, genera documentos, calcula embeddings, construye un índice FAISS local y solo al final publica los artefactos en Cloud Storage. Esto implica que, si el job falla antes de la publicación, el trabajo parcial no queda disponible como índice consumible por el backend.

Las principales limitaciones actuales son:

```text
1. Reindexación completa
   Cada corrida full vuelve a procesar todos los registros seleccionados.

2. No hay manejo incremental
   Altas, bajas y modificaciones de propiedades no se procesan como deltas.

3. No hay checkpoint persistente real
   Los checkpoints locales viven en /tmp y se pierden si el contenedor muere.

4. Publicación al final del proceso
   El índice solo aparece en GCS cuando todo el pipeline termina correctamente.

5. Dependencia directa de Gemini Embeddings
   El costo y el tiempo dependen de llamadas externas al modelo de embeddings.

6. Escalamiento vertical
   La optimización actual se basa principalmente en subir CPU/memoria del job.

7. Sin particionamiento del índice
   Se construye un único FAISS index para la colección completa.

8. Sin gestión de disponibilidad de propiedades
   Propiedades dadas de baja siguen presentes hasta que BigQuery y FAISS se reconstruyan.

9. Sin monitoreo de drift
   No hay alerta automática cuando cambia mucho la distribución de propiedades, precios o barrios.

10. Sin estrategia formal de rollback automático
   Existen rutas versionadas, pero el backend consume latest salvo intervención manual.
```

## Consideraciones para industrialización

Para industrializar el `job-indexer`, el siguiente paso no sería necesariamente usar Spark. El volumen actual no justifica una plataforma distribuida pesada; el problema principal se puede atacar mejor con una arquitectura incremental, particionada y orientada a artefactos.

Una versión más robusta debería separar el proceso en unidades más pequeñas:

```text
BigQuery changes / snapshot
→ delta detection
→ document builder
→ embedding batch worker
→ partial vector index builder
→ index merger / publisher
→ manifest versionado
→ backend refresh controlado
```

## Manejo de altas, bajas y modificaciones

Actualmente el índice se reconstruye completo. Para producción, conviene agregar una tabla o vista de control con estado por propiedad:

```text
listing_id
source_updated_at
content_hash
status
is_active
last_indexed_at
embedding_model
embedding_version
index_version
```

Con esto se pueden detectar tres casos:

```text
Altas:
  listing_id nuevo que no existe en el índice anterior.

Modificaciones:
  listing_id existente cuyo content_hash cambió.

Bajas:
  listing_id que ya no aparece activo o cuyo status indica no disponible.
```

El `content_hash` debería calcularse sobre el texto que realmente se embebe, por ejemplo:

```text
title_clean + description_clean + barrio_fixed + price + amenities + features
```

Así se evita re-embebber propiedades que no cambiaron semánticamente.

## Estrategia de deltas

Una estrategia incremental podría usar BigQuery como fuente de verdad y una tabla de control de indexación:

```text
real_estate_listings
→ generar content_hash actual
→ comparar contra index_manifest/listing_index_state
→ obtener changed_ids
→ embebber solo changed_ids
→ actualizar índice parcial o reconstruir segmentos afectados
```

Ejemplo conceptual:

```sql
SELECT
  current.id
FROM current_listings current
LEFT JOIN indexed_state previous
  ON current.id = previous.id
WHERE previous.id IS NULL
   OR current.content_hash != previous.content_hash
   OR current.status != previous.status
```

Para bajas, se puede mantener una lista de exclusión en el manifest o reconstruir periódicamente el índice completo.

## Paralelización sin Spark

Para este tamaño de datos, una opción más natural que Spark sería dividir el proceso por particiones y usar Cloud Run Jobs paralelos o Tasks. La idea sería partir los registros en segmentos independientes:

```text
partition 0 → ids 0000-0499
partition 1 → ids 0500-0999
partition 2 → ids 1000-1499
...
```

Cada partición genera embeddings y un índice FAISS parcial:

```text
gs://.../faiss/realstate_mvd/versions/<version>/parts/part-000/
  index.faiss
  index.pkl
  listing_ids.json
  manifest.json
```

Luego un paso final de consolidación puede:

```text
1. Validar que todas las partes terminaron.
2. Crear un manifest global.
3. Publicar latest.
4. Notificar al backend que hay nueva versión disponible.
```

Esto permitiría bajar tiempos sin depender únicamente de más CPU/memoria en una sola instancia.

## Servicios o componentes a tocar

Para evolucionar el prototipo hacia una versión industrializada, habría que modificar o agregar estas piezas:

```text
apps/job-indexer/app/services/bigquery_reader.py
  Agregar lectura por partición, filtros por delta, content_hash y estados.

apps/job-indexer/app/services/listing_document_service.py
  Garantizar construcción determinística del texto embebido.

apps/job-indexer/app/services/embedding_service.py
  Soportar batches persistentes, checkpoints en GCS y reintentos controlados.

apps/job-indexer/app/services/faiss_builder.py
  Construir índices parciales y registrar metadatos por partición.

apps/job-indexer/app/services/gcs_service.py
  Publicar partes, manifests parciales y manifests globales.

apps/job-indexer/app/build_index.py
  Orquestar modo full, modo delta y modo partition.

shared/python/miad_rag_common/schemas
  Agregar contratos para manifest, index version, partition metadata y estado de indexación.

infra/envs/dev/main.tf
  Parametrizar CPU, memoria, timeout, retries, BQ_LIMIT, particiones y modo de ejecución.

apps/backend/app/services/gcs_index_service.py
  Leer manifest global, validar versión y eventualmente refrescar latest.

apps/backend/app/services/retrieval_service.py
  Prepararse para consultar múltiples índices parciales si se decide no consolidar.
```

## Modos futuros sugeridos

El job podría soportar tres modos de ejecución:

```text
FULL
  Reconstruye todo el índice desde BigQuery.

DELTA
  Procesa solo registros nuevos o modificados.

PARTITION
  Construye una partición específica del índice.

MERGE
  Consolida o publica una versión global a partir de partes.
```

Ejemplo de variables futuras:

```hcl
INDEX_BUILD_MODE = "FULL"      # FULL | DELTA | PARTITION | MERGE
INDEX_PARTITION_ID = "0"
INDEX_PARTITION_COUNT = "8"
INDEX_VERSION = "20260504T052700Z"
BQ_LIMIT = null
MAX_RETRIES = 0
```

## Recomendación de siguiente etapa

Para el prototipo académico, la configuración `4 CPU / 8Gi` con corrida full en 26.18 minutos es aceptable como baseline. Para industrializar, la prioridad no debería ser subir indefinidamente CPU, sino reducir trabajo repetido mediante deltas, persistir checkpoints fuera del contenedor, particionar el índice y publicar manifests versionados que permitan trazabilidad, rollback y actualización controlada del backend.
