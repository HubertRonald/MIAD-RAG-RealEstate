<p align="center">
    <img src="./figs/Banner_PAAD_01.jpg" width="980" />
</p>

<p align="left">
    <!-- Infraestructura y herramientas base -->
    <a href="https://cloud.google.com" target="_blank">
    <img src="https://img.shields.io/badge/-Google%20Cloud%20Platform-4285F4?style=flat-square&logo=google%20cloud&logoColor=white" />
    </a>
    <a href="https://hub.docker.com/r/google/cloud-sdk" target="_blank">
        <img src="https://img.shields.io/badge/docker-%230db7ed.svg?style=flat-square&logo=docker&logoColor=white" />
    </a>
    <a href="https://code.visualstudio.com/download" target="_blank">
    <img src="https://img.shields.io/badge/Visual%20Studio%20Code-007ACC?style=flat-square&logo=visualstudiocode&logoColor=fff" />
    </a>
    <!-- Testing y automatización -->
  <a href="https://docs.pytest.org/en/stable/" target="_blank">
    <img src="https://img.shields.io/badge/pytest-0A9EDC?style=flat-square&logo=pytest&logoColor=white" />
  </a>
  <a href="https://tox.wiki/en/latest/" target="_blank">
    <img src="https://img.shields.io/badge/tox-20C997?style=flat-square&logo=tox&logoColor=white" />
  </a>
    <!-- API y backend -->
  <a href="https://fastapi.tiangolo.com/" target="_blank">
    <img src="https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white" />
  </a>
  <a href="https://www.uvicorn.org/" target="_blank">
    <img src="https://img.shields.io/badge/Uvicorn-111827?style=flat-square&logo=uvicorn&logoColor=white" />
  </a>
  <a href="https://pydantic.dev/" target="_blank">
    <img src="https://img.shields.io/badge/Pydantic-E92063?style=flat-square&logo=pydantic&logoColor=white" />
  </a>
  <a href="https://github.com/Delgan/loguru" target="_blank">
    <img src="https://img.shields.io/badge/Loguru-EE4C2C?style=flat-square&logo=loguru&logoColor=white" />
  </a>
  <a href="https://docs.python.org/3/library/typing.html" target="_blank">
    <img src="https://img.shields.io/badge/typing_extensions-3776AB?style=flat-square&logo=python&logoColor=white" />
  </a>
  <a href="https://pypi.org/project/python-multipart/" target="_blank">
    <img src="https://img.shields.io/badge/python--multipart-2C5BB4?style=flat-square&logo=python&logoColor=white" />
  </a>
    <!-- Lenguaje y librerías de ML -->
    <a href="https://peps.python.org/pep-0596/#schedule-first-bugfix-release" target="_blank">
        <img src="https://img.shields.io/badge/python-3670A0?style=flat-square&logo=python&logoColor=ffdd54" />
    </a>
    </a>
    <a href="https://pandas.pydata.org/" target="_blank">
        <img src="https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white" />
    </a>
    <a href="https://numpy.org/" target="_blank">
        <img src="https://img.shields.io/badge/numpy-013243?style=flat-square&logo=numpy&logoColor=white" />
    </a>
    <!-- CI/CD e infraestructura -->
  <a href="https://yaml.org/" target="_blank">
    <img src="https://img.shields.io/badge/YAML-CB171E?style=flat-square&logo=yaml&logoColor=white" />
  </a>
  <a href="https://www.json.org/json-en.html" target="_blank">
    <img src="https://img.shields.io/badge/json-5E5C5C?style=flat-square&logo=json&logoColor=white" />
  </a>
  <a href="https://pypi.org/project/setuptools/" target="_blank">
    <img src="https://img.shields.io/badge/pkg-3776AB?style=flat-square&logo=pypi&logoColor=white" />
  </a>
     <!-- Actividad del repositorio -->
    <img src="https://img.shields.io/github/last-commit/HubertRonald/MIAD-RAG-RealEstate?style=flat-square" />
    <img src="https://img.shields.io/github/commit-activity/t/HubertRonald/MIAD-RAG-RealEstate?style=flat-square&color=dodgerblue" />
</p>

# MIAD-RAG-RealEstate
### RAG-based Real Estate Recommendation System on GCP  
**Montevideo, Uruguay**


## Resumen

Sistema de recomendación inmobiliaria basado en **Retrieval-Augmented Generation (RAG)** que permite a los usuarios buscar propiedades mediante lenguaje natural, combinando:

- Búsqueda semántica (FAISS)
- Enriquecimiento estructurado (BigQuery)
- Generación de explicaciones (LLM)
- Visualización geográfica (Streamlit)


## Arquitectura GCP

<p align="center">
    <img src="./figs/MIAD-RAG-RealEstate.png" width="980" />
</p>

**Stack principal:**

- Cloud Run (Frontend + Backend)
- BigQuery (datos estructurados)
- Cloud Storage (FAISS backup)
- Secret Manager (seguridad)
- Gemini API (LLM + embeddings)
  

## Flujo de Solución (RAG Pipeline)

```mermaid
%%{init: {
  "theme": "default",
  "flowchart": { "curve": "basis", "nodeSpacing": 50, "rankSpacing": 70 }
}}%%
flowchart LR

%% ======================
%% FUERA DE GCP
%% ======================
subgraph EXT["Usuario"]
    U[Usuario]
    BROWSER[Navegador]
end

%% ======================
%% FRONTEND
%% ======================
subgraph FE["Experiencia de Usuario"]
    UI[Cloud Run<br/>Streamlit App<br/>Busqueda + Mapa + Cards]
end

%% ======================
%% BACKEND
%% ======================
subgraph BE["Motor RAG (Orquestación)"]
    API[Cloud Run<br/>FastAPI]
end

%% ======================
%% DATA LAYER
%% ======================
subgraph DATA["Datos"]
    FAISS[FAISS Index<br/>in-memory]
    GCS[Cloud Storage<br/>Backup indice]
    BQ[BigQuery<br/>Datos inmobiliarios]
end

%% ======================
%% AI SERVICES
%% ======================
subgraph AI["Inteligencia Artificial"]
    GEM[Gemini API<br/>Generación + Embeddings]
end

%% ======================
%% SEGURIDAD
%% ======================
subgraph SEC["Seguridad"]
    SM[Secret Manager<br/>Credenciales seguras]
end

%% ======================
%% OBSERVABILIDAD
%% ======================
subgraph OBS["Monitoreo y Evaluación"]
    LOGS[Cloud Logging]
    TRACE[LangSmith / RAGAS]
end

%% ======================
%% OFFLINE
%% ======================
subgraph OFF["Pipeline Offline"]
    SCRAPER[ExploracionDatos.py]
    FEAT[Feature Engineering]
    EMB[Embeddings]
    BUILD[Construcción FAISS]
end

%% ======================
%% FLUJO PRINCIPAL
%% ======================
U --> BROWSER --> UI --> API

API --> FAISS
GCS --> FAISS

FAISS --> IDS[Top-K propiedades similares]
IDS --> BQ
BQ --> DATA2[Datos enriquecidos<br/>precio, ubicación, atributos]

DATA2 --> API
API --> GEM
GEM --> RESP[Explicación + recomendación]

RESP --> UI --> BROWSER

%% ======================
%% SEGURIDAD
%% ======================
API --> SM
GEM --> SM

%% ======================
%% OBSERVABILIDAD
%% ======================
API -.-> LOGS
API -.-> TRACE

%% ======================
%% OFFLINE FLOW
%% ======================
SCRAPER --> FEAT --> EMB --> BUILD --> GCS
FEAT --> BQ

%% ======================
%% ESTILOS
%% ======================
classDef fe fill:#8AB4F8,color:#fff,stroke:#5A95F5,stroke-width:2px
classDef be fill:#4285F4,color:#fff,stroke:#3367D6,stroke-width:2px
classDef data fill:#81C995,color:#fff,stroke:#4CAF50,stroke-width:2px
classDef ai fill:#34A853,color:#fff,stroke:#0F9D58,stroke-width:2px
classDef sec fill:#FDD663,color:#000,stroke:#F9AB00,stroke-width:2px
classDef obs fill:#F28B82,color:#fff,stroke:#D93025,stroke-width:2px
classDef off fill:#C58AF9,color:#fff,stroke:#9334E6,stroke-width:2px

class UI fe
class API be
class FAISS,GCS,BQ data
class GEM ai
class SM sec
class LOGS,TRACE obs
class SCRAPER,FEAT,EMB,BUILD off

style EXT fill:#F1F3F4,stroke:#9AA0A6
style FE fill:#F1F3F4,stroke:#9AA0A6
style BE fill:#F1F3F4,stroke:#9AA0A6
style DATA fill:#F1F3F4,stroke:#9AA0A6
style AI fill:#F1F3F4,stroke:#9AA0A6
style SEC fill:#F1F3F4,stroke:#9AA0A6
style OBS fill:#F1F3F4,stroke:#9AA0A6
style OFF fill:#F1F3F4,stroke:#9AA0A6
```

## .gitignore

Fue generado en [gitignore.io](https://www.toptal.com/developers/gitignore/) con los filtros `python`, `macos`, `windows` y consumido mediante su API como archivo crudo desde la terminal:

```bash
curl -L https://www.toptal.com/developers/gitignore/api/python,macos,windows > .gitignore
```

## Shields, Links

Los shields en las cabeceras de este `Readme.md` se generaron con:

- <a href="https://shields.io/" target="_blank"><span>https://shields.io/</span></a>
- <a href="https://github.com/inttter/md-badges" target="_blank"><span>https://github.com/inttter/md-badges</span></a>

> **NOTA:** Todos los shields y/o enlaces cuando se imprima este `Readme.md` a `.pdf` puede ser usados haciendo `Ctrl + Clic` (windows) or `Cmd + Clic` (macOS) sobre los mismos.

## Licencia y derechos de autor

El código fuente de este proyecto se distribuye bajo licencia MIT - ver la [LICENCIA](LICENSE) archivo (en inglés) para más detalle.

En caso de utilizar materiales con derechos reservados, estos se emplean únicamente para fines de **investigación, análisis y demostración académica**, sin fines comerciales.