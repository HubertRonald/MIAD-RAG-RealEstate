"""
Módulo principal de la aplicación FastAPI

Este módulo configura y lanza la aplicación FastAPI principal,
integrando todos los routers y configurando middleware para CORS.
"""

# ==================== CARGAR VARIABLES DE ENTORNO ====================
# IMPORTANTE: Debe ser lo primero para que las credenciales estén disponibles
from dotenv import load_dotenv
import os

# Cargar variables de entorno desde .env
load_dotenv()

# Verificar que GOOGLE_API_KEY esté configurada
if not os.getenv("GOOGLE_API_KEY"):
    print("WARNING: GOOGLE_API_KEY no encontrada en variables de entorno")
    print("Por favor, configura tu API key en el archivo .env")
else:
    print("GOOGLE_API_KEY cargada correctamente")

# ==================== IMPORTS DE FASTAPI ====================

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers.load_from_url import router as load_from_url_router
from app.routers.load_from_csv import router as load_from_csv_router
from app.routers.ask import router as ask_router
from app.routers.health import router as health_router
from app.routers.validate_load import router as validate_load_router

# ==================== CONFIGURACIÓN DE LA APLICACIÓN ====================

# Instancia principal de FastAPI
app = FastAPI()

# Orígenes permitidos para CORS (desarrollo local)
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://misw4411-frontend-33.web.app",
]

# ==================== CONFIGURACIÓN DE MIDDLEWARE ====================

# Middleware CORS para permitir peticiones desde frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],     # Todos los métodos HTTP
    allow_headers=["*"],     # Todos los headers (Content-Type, Accept, etc.)
)

# ==================== INTEGRACIÓN DE ROUTERS ====================

# Incluir todos los routers del sistema
app.include_router(load_from_url_router)
app.include_router(load_from_csv_router)
app.include_router(ask_router)
app.include_router(health_router)
app.include_router(validate_load_router)

# ==================== ENDPOINTS PRINCIPALES ====================

@app.get("/")
def read_root():
    """
    Endpoint raíz de la API.
    
    Proporciona un mensaje de bienvenida y confirma que la API
    está funcionando correctamente.
    
    Returns:
        dict: Mensaje de bienvenida con información básica
    """
    return {"message": "Bienvenido a la API de carga de documentos"}


# ==================== EVENTOS DE APLICACIÓN ====================

@app.on_event("startup")
async def startup_event():
    """
    Evento de inicio de la aplicación.
    
    Se ejecuta cuando la aplicación FastAPI se inicia.
    """
    print("Servidor iniciado correctamente")