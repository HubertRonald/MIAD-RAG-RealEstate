from __future__ import annotations
from typing import Optional
from pydantic import BaseModel


class MapPoint(BaseModel):
    """
    Punto geográfico simplificado para renderizar mapa en Streamlit.

    Este modelo NO viene del contrato original de _realstate_ragas.
    Es una extensión del nuevo frontend.
    """

    id: str
    lat: float
    lon: float
    label: Optional[str] = None
    barrio: Optional[str] = None
    price_fixed: Optional[float] = None
    currency_fixed: Optional[str] = None
    match_score: Optional[int] = None
    rank: Optional[int] = None
