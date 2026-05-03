# Construcción del Índice FAISS en modo Desarrollo

El indice FAISS almacena las publicaciones de inmuebles como vectores númericos que los sistemas de aprendizaje de lenguaje natural como los LLMs puedan interpretar. 

## Prerrequisitos

- Contenedor Docker en ejecución (`docker compose up`)
- `listings.csv` ubicado en `docs/realstate_mvd/listings.csv`
- `GOOGLE_API_KEY` configurada en `.env` (debe pertenecer a un proyecto de Google Cloud con facturación habilitada)

---

## 1. Abrir el notebook de validación

Abrir `realstate_ragas/local_tests/validate_faiss_load.ipynb` y configurar el directorio de trabajo en la raíz del repositorio:

```python
import os
ROOT ="/ruta/al/repositorio/realstate_ragas"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

print(f"Working directory: {os.getcwd()}")
```

---

## 2. Previsualizar los documentos antes de indexar

Ejecutar una verificación para confirmar que el CSV se está parseando correctamente antes de realizar el proceso completo de embeddings (~3–4 minutos + costo de API):

```python
from app.services.csv_document_service import CSVDocumentService

svc = CSVDocumentService()

# Verificar segmentos disponibles
segments = svc.get_available_segments("./docs/realstate_mvd/listings.csv")
print(segments)
# Esperado: ~3.377 listings, 4 combinaciones de tipo de operación/propiedad, ~50 barrios

# Previsualizar los primeros 3 documentos
svc.preview_document("./docs/realstate_mvd/listings.csv", n=3)
```

Verificar que:
- `Ubicación:` aparece en los listings que tienen datos de dirección (`l3`)
- `planta baja` se renderiza correctamente para piso 0
- `a estrenar` aparece en propiedades nuevas (`age=0` o `condition=new`)
- Ningún listing muestra más de 3–4 cocheras
- Amenities como `piscina`, `terraza`, `parrillero` aparecen donde corresponde
- `Disponible para venta y alquiler.` aparece en listings con `is_dual_intent=True`
- Los listings con `barrio_confidence='marketing_inflation'` usan `title_clean` y `desc_clean` en lugar del título y descripción originales
- Las descripciones largas se truncan a 5.000 caracteres (el texto termina con `...`); en la práctica ningún listing del corpus actual alcanza este límite

---

## 3. Construir el índice

```python
from app.services.load_documents_service import load_from_csv

result = await load_from_csv("realstate_mvd")

print(f"Éxito:                {result['success']}")
print(f"Mensaje:              {result['message']}")
print(f"Documentos indexados: {result['data']['processing_summary']['total_documents']}")
print(f"Costo estimado:       ${result['data']['embedding_statistics']['cost_breakdown']['embedding_cost_usd']}")
```

Salida esperada:
```
Éxito:                True
Mensaje:              Colección 'realstate_mvd' indexada exitosamente desde CSV
Documentos indexados: 3377
Costo estimado:       ~$0.027
```

### Límites de tasa (rate limits)

El servicio de embeddings utiliza procesamiento por lotes (50 docs/lote) con retroceso exponencial ante errores 429. Si el proceso se interrumpe:

- **No reiniciar desde cero** — se guarda un checkpoint tras cada lote en `faiss_index/realstate_mvd/_embedding_checkpoint.json`
- Simplemente volver a ejecutar `await load_from_csv("realstate_mvd")` y el proceso retomará desde el último lote completado
- El checkpoint se elimina automáticamente una vez que el índice está completo

> **Nota:** Asegurarse de que `GOOGLE_API_KEY` pertenezca a un proyecto con facturación habilitada. Las cuotas del tier gratuito (1.000 RPD) son insuficientes para construir el índice completo.

---

## 4. Validar el índice

```python
from app.services.embedding_service import EmbeddingService
from app.services.retrieval_service import RetrievalService, PropertyFilters

# Cargar el índice
emb = EmbeddingService()
emb.load_vectorstore("./faiss_index/realstate_mvd")
ret = RetrievalService(emb, k=5)

# Test 1 — búsqueda semántica (sin filtros)
docs = ret.retrieve_documents("apartamento con piscina cerca del mar")
print(f"Resultados búsqueda semántica: {len(docs)}")
for d in docs:
    print(d.page_content[:150])
    print()

# Test 2 — búsqueda con filtros
filters = PropertyFilters(
    operation_type="venta",
    barrio="POCITOS",
    max_price=200000
)
docs = ret.retrieve_with_filters("apartamento luminoso", filters)
print(f"Resultados búsqueda filtrada: {len(docs)}")
for d in docs:
    relevant = {k: v for k, v in d.metadata.items()
                if k in ["barrio_fixed", "operation_type", "price_fixed",
                         "barrio_confidence", "is_dual_intent"]}
    print(relevant)
```

Ambos tests deben retornar resultados. Si la búsqueda filtrada retorna 0, aumentar `fetch_k`:

```python
ret = RetrievalService(emb, k=5, fetch_k=200)
```

---

## 5. Ubicación del índice

El índice construido se guarda en:
```
faiss_index/
└── realstate_mvd/
    ├── index.faiss
    └── index.pkl
```

Este directorio está montado como volumen Docker y persiste entre reinicios del contenedor. **No es necesario reconstruirlo** a menos que cambien los datos del CSV.

---

## Reconstrucción del índice

Si `listings.csv` se actualiza (nuevos datos, correcciones de bugs, cambios de columnas), el índice debe reconstruirse:

1. Eliminar el índice existente: `rm -rf faiss_index/realstate_mvd/`
2. Repetir los pasos 2–4 anteriores
