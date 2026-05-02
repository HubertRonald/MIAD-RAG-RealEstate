from __future__ import annotations
from typing import Any, Callable, Optional, Union
from pydantic import BaseModel


class PropertyFilters(BaseModel):
    """
    Filtros estructurados para búsqueda de propiedades.

    Estos filtros sirven para:
    - construir filtros explícitos desde el request del frontend,
    - enriquecer filtros con PreferenceExtractionService,
    - aplicar pre-filtrado de metadata antes de la búsqueda semántica en FAISS,
    - serializar filters_applied en la respuesta del backend.
    """

    # Segmentación
    operation_type: Optional[str] = None
    property_type: Optional[str] = None
    barrio: Optional[Union[str, list[str]]] = None

    # Precio
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    max_price_m2: Optional[float] = None

    # Características físicas
    min_bedrooms: Optional[int] = None
    max_bedrooms: Optional[int] = None
    min_surface: Optional[float] = None
    max_surface: Optional[float] = None

    # Entorno urbano
    max_dist_plaza: Optional[float] = None
    max_dist_playa: Optional[float] = None

    # Amenities
    has_pool: bool = False
    has_gym: bool = False
    has_elevator: bool = False
    has_parrillero: bool = False
    has_terrace: bool = False
    has_rooftop: bool = False
    has_security: bool = False
    has_storage: bool = False
    has_parking: bool = False
    has_party_room: bool = False
    has_green_area: bool = False
    has_playground: bool = False
    has_visitor_parking: bool = False


    def active_dict(self) -> dict[str, Any]:
        """
        Retorna solo los filtros realmente activos.

        Útil para responder filters_applied al frontend sin llenar el JSON
        con campos null o flags false.
        """
        result: dict[str, Any] = {}

        for field_name in [
            "operation_type",
            "property_type",
            "barrio",
            "min_price",
            "max_price",
            "max_price_m2",
            "min_bedrooms",
            "max_bedrooms",
            "min_surface",
            "max_surface",
            "max_dist_plaza",
            "max_dist_playa",
        ]:
            value = getattr(self, field_name)
            if value is not None:
                result[field_name] = value

        for field_name in [
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
        ]:
            if getattr(self, field_name):
                result[field_name] = True

        return result

    def is_empty(self) -> bool:
        """True si no hay ningún filtro activo."""
        return not bool(self.active_dict())
    
    @staticmethod
    def _as_float(v: Any) -> Optional[float]:
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _as_int(v: Any) -> Optional[int]:
        if v is None:
            return None
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _is_true(v: Any) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return v == 1
        if isinstance(v, str):
            return v.strip().lower() in {"1","true","yes","y","si","sí"}
        return False
    
    def to_filter_fn(self) -> Optional[Callable[[dict[str, Any]], bool]]:
        """
        Convierte los filtros en un callable compatible con FAISS/LangChain.

        La función recibe metadata de un documento y retorna True/False.
        Se usa para filtros con rangos numéricos y lógica AND.
        """
        if self.is_empty():
            return None

        filters = self

        if self.is_empty():
            return None
        filters = self
        def _filter(meta: dict[str, Any]) -> bool:
            if filters.operation_type and meta.get("operation_type") != filters.operation_type:
                return False
            if filters.property_type and meta.get("property_type") != filters.property_type:
                return False
            if filters.barrio:
                barrios = [filters.barrio] if isinstance(filters.barrio, str) else filters.barrio
                if (meta.get("barrio_fixed") or meta.get("barrio")) not in barrios:
                    return False
            price = self._as_float(meta.get("price_fixed"))
            if price is not None:
                if filters.min_price is not None and price < filters.min_price:
                    return False
                if filters.max_price is not None and price > filters.max_price:
                    return False
            pm2 = self._as_float(meta.get("price_m2"))
            if filters.max_price_m2 is not None and pm2 is not None and pm2 > filters.max_price_m2:
                return False
            bedrooms = self._as_int(meta.get("bedrooms"))
            if bedrooms is not None:
                if filters.min_bedrooms is not None and bedrooms < filters.min_bedrooms:
                    return False
                if filters.max_bedrooms is not None and bedrooms > filters.max_bedrooms:
                    return False
            surface = self._as_float(meta.get("surface_covered")) or self._as_float(meta.get("surface_total"))
            if surface is not None:
                if filters.min_surface is not None and surface < filters.min_surface:
                    return False
                if filters.max_surface is not None and surface > filters.max_surface:
                    return False
            dist_plaza = self._as_float(meta.get("dist_plaza"))
            if filters.max_dist_plaza is not None and dist_plaza is not None and dist_plaza > filters.max_dist_plaza:
                return False
            dist_playa = self._as_float(meta.get("dist_playa"))
            if filters.max_dist_playa is not None and dist_playa is not None and dist_playa > filters.max_dist_playa:
                return False
            for flag in [
                "has_pool","has_gym","has_elevator","has_parrillero","has_terrace","has_rooftop",
                "has_security","has_storage","has_party_room","has_green_area","has_playground","has_visitor_parking",
            ]:
                if getattr(filters, flag) and not self._is_true(meta.get(flag)):
                    return False
            if filters.has_parking:
                garages = self._as_float(meta.get("garages"))
                if garages is None or garages <= 0:
                    return False
            return True
        return _filter
    
    