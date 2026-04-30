from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from miad_rag_common.schemas.filters import PropertyFilters
from miad_rag_common.utils.norm_barrio_utils import normalize_barrio

logger = logging.getLogger(__name__)

_BOOL_FIELDS = frozenset(
    {
        "has_pool",
        "has_gym",
        "has_elevator",
        "has_parrillero",
        "has_terrace",
        "has_rooftop",
        "has_security",
        "has_storage",
        "has_parking",
        "has_party_room",
        "has_green_area",
        "has_playground",
        "has_visitor_parking",
    }
)

_FLOAT_FIELDS = frozenset(
    {
        "min_price",
        "max_price",
        "max_price_m2",
        "min_surface",
        "max_surface",
        "max_dist_plaza",
        "max_dist_playa",
    }
)

_INT_FIELDS = frozenset({"min_bedrooms", "max_bedrooms"})

_VALID_OPERATION_TYPES = {"venta", "alquiler"}
_VALID_PROPERTY_TYPES = {"apartamentos", "casas"}

_EXTRACTION_SYSTEM = """
Eres un extractor de preferencias para un sistema inmobiliario de Montevideo, Uruguay.

Tu única tarea es extraer preferencias de búsqueda como JSON.

REGLAS:
- Responde SOLO con un objeto JSON válido.
- No uses markdown.
- Usa null para campos no mencionados.
- No inventes valores.
- Los filtros explícitos del sistema tendrán precedencia después.

Campos:
{
  "operation_type": "venta" | "alquiler" | null,
  "property_type": "apartamentos" | "casas" | null,
  "barrio": string | null,
  "min_price": number | null,
  "max_price": number | null,
  "max_price_m2": number | null,
  "min_bedrooms": integer | null,
  "max_bedrooms": integer | null,
  "min_surface": number | null,
  "max_surface": number | null,
  "max_dist_plaza": number | null,
  "max_dist_playa": number | null,
  "has_pool": boolean,
  "has_gym": boolean,
  "has_elevator": boolean,
  "has_parrillero": boolean,
  "has_terrace": boolean,
  "has_rooftop": boolean,
  "has_security": boolean,
  "has_storage": boolean,
  "has_parking": boolean,
  "has_party_room": boolean,
  "has_green_area": boolean,
  "has_playground": boolean,
  "has_visitor_parking": boolean
}

Ejemplos:
Input: "busco apto de 2 dormitorios en pocitos hasta 150 mil con piscina"
Output: {"property_type":"apartamentos","barrio":"POCITOS","max_price":150000,"min_bedrooms":2,"max_bedrooms":2,"has_pool":true}

Input: "algo cerca del mar con terraza"
Output: {"max_dist_playa":500,"has_terrace":true}
"""


class PreferenceExtractionService:
    """
    Extrae PropertyFilters desde texto libre y los combina con filtros explícitos.
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.0,
    ) -> None:
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
        )

    def extract(
        self,
        question: str,
        explicit_filters: PropertyFilters | None = None,
    ) -> PropertyFilters:
        explicit = explicit_filters or PropertyFilters()
        llm_data = self._extract_from_llm(question)

        return PropertyFilters(
            operation_type=explicit.operation_type or llm_data.get("operation_type"),
            property_type=explicit.property_type or llm_data.get("property_type"),
            barrio=explicit.barrio or llm_data.get("barrio"),
            min_price=explicit.min_price
            if explicit.min_price is not None
            else llm_data.get("min_price"),
            max_price=explicit.max_price
            if explicit.max_price is not None
            else llm_data.get("max_price"),
            max_price_m2=explicit.max_price_m2
            if explicit.max_price_m2 is not None
            else llm_data.get("max_price_m2"),
            min_bedrooms=explicit.min_bedrooms
            if explicit.min_bedrooms is not None
            else llm_data.get("min_bedrooms"),
            max_bedrooms=explicit.max_bedrooms
            if explicit.max_bedrooms is not None
            else llm_data.get("max_bedrooms"),
            min_surface=explicit.min_surface
            if explicit.min_surface is not None
            else llm_data.get("min_surface"),
            max_surface=explicit.max_surface
            if explicit.max_surface is not None
            else llm_data.get("max_surface"),
            max_dist_plaza=explicit.max_dist_plaza
            if explicit.max_dist_plaza is not None
            else llm_data.get("max_dist_plaza"),
            max_dist_playa=explicit.max_dist_playa
            if explicit.max_dist_playa is not None
            else llm_data.get("max_dist_playa"),
            has_pool=explicit.has_pool or llm_data.get("has_pool", False),
            has_gym=explicit.has_gym or llm_data.get("has_gym", False),
            has_elevator=explicit.has_elevator or llm_data.get("has_elevator", False),
            has_parrillero=explicit.has_parrillero or llm_data.get("has_parrillero", False),
            has_terrace=explicit.has_terrace or llm_data.get("has_terrace", False),
            has_rooftop=explicit.has_rooftop or llm_data.get("has_rooftop", False),
            has_security=explicit.has_security or llm_data.get("has_security", False),
            has_storage=explicit.has_storage or llm_data.get("has_storage", False),
            has_parking=explicit.has_parking or llm_data.get("has_parking", False),
            has_party_room=explicit.has_party_room or llm_data.get("has_party_room", False),
            has_green_area=explicit.has_green_area or llm_data.get("has_green_area", False),
            has_playground=explicit.has_playground or llm_data.get("has_playground", False),
            has_visitor_parking=explicit.has_visitor_parking
            or llm_data.get("has_visitor_parking", False),
        )

    def _extract_from_llm(self, question: str) -> dict[str, Any]:
        if not question or not question.strip():
            return {}

        try:
            messages = [
                SystemMessage(content=_EXTRACTION_SYSTEM),
                HumanMessage(
                    content=(
                        f"Texto del usuario: {question}\n\n"
                        "Extrae las preferencias como JSON."
                    )
                ),
            ]

            response = self.llm.invoke(messages)
            return self._parse_and_validate(response.content.strip())

        except Exception as exc:
            logger.warning("[PreferenceExtraction] fallo extracción: %s", exc)
            return {}

    def _parse_and_validate(self, raw_text: str) -> dict[str, Any]:
        clean = re.sub(r"```(?:json)?\s*|```", "", raw_text).strip()

        try:
            data = json.loads(clean)
        except json.JSONDecodeError:
            return {}

        if not isinstance(data, dict):
            return {}

        result: dict[str, Any] = {}

        op = data.get("operation_type")
        if isinstance(op, str) and op.lower() in _VALID_OPERATION_TYPES:
            result["operation_type"] = op.lower()

        property_type = data.get("property_type")
        if isinstance(property_type, str) and property_type.lower() in _VALID_PROPERTY_TYPES:
            result["property_type"] = property_type.lower()

        barrio = data.get("barrio")
        if isinstance(barrio, str) and barrio.strip():
            normalized = normalize_barrio(barrio)
            if normalized:
                result["barrio"] = normalized

        for field in _FLOAT_FIELDS:
            value = data.get(field)
            if value is not None:
                try:
                    coerced = float(value)
                    if coerced > 0:
                        result[field] = coerced
                except (TypeError, ValueError):
                    continue

        for field in _INT_FIELDS:
            value = data.get(field)
            if value is not None:
                try:
                    result[field] = int(value)
                except (TypeError, ValueError):
                    continue

        for field in _BOOL_FIELDS:
            if data.get(field) is True:
                result[field] = True

        return result
