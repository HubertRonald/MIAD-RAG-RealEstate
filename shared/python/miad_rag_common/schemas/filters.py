from __future__ import annotations

from typing import Any, Callable, ClassVar, Optional, Union

from pydantic import BaseModel, ConfigDict

from miad_rag_common.utils.text_utils import (
    normalize_for_match,
    safe_bool,
    safe_float,
    safe_int,
)


class PropertyFilters(BaseModel):
    """
    Filtros estructurados para búsqueda de propiedades.

    Estos filtros se usan para:

    - construir filtros explícitos desde el request del frontend;
    - enriquecer filtros con PreferenceExtractionService;
    - aplicar pre-filtrado de metadata antes de búsqueda semántica en FAISS;
    - serializar filters_applied en la respuesta del backend.

    Los filtros se combinan con AND lógico.

    Nota:
        Este modelo vive en shared porque no pertenece únicamente al backend.
        Representa el contrato interno de filtrado estructurado para listings.
    """

    model_config = ConfigDict(
        extra="ignore",
        arbitrary_types_allowed=True,
    )

    # -------------------------------------------------------------------------
    # Segmentación
    # -------------------------------------------------------------------------
    operation_type: Optional[str] = None
    property_type: Optional[str] = None
    barrio: Optional[Union[str, list[str]]] = None

    # -------------------------------------------------------------------------
    # Precio
    # -------------------------------------------------------------------------
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    max_price_m2: Optional[float] = None

    # -------------------------------------------------------------------------
    # Características físicas
    # -------------------------------------------------------------------------
    min_bedrooms: Optional[int] = None
    max_bedrooms: Optional[int] = None
    min_surface: Optional[float] = None
    max_surface: Optional[float] = None

    # -------------------------------------------------------------------------
    # Entorno urbano
    # -------------------------------------------------------------------------
    max_dist_plaza: Optional[float] = None
    max_dist_playa: Optional[float] = None

    # -------------------------------------------------------------------------
    # Amenities
    # -------------------------------------------------------------------------
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

    EXACT_FIELDS: ClassVar[list[str]] = [
        "operation_type",
        "property_type",
        "barrio",
    ]

    RANGE_FIELDS: ClassVar[list[str]] = [
        "min_price",
        "max_price",
        "max_price_m2",
        "min_bedrooms",
        "max_bedrooms",
        "min_surface",
        "max_surface",
        "max_dist_plaza",
        "max_dist_playa",
    ]

    FLAG_FIELDS: ClassVar[list[str]] = [
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
    ]

    # =========================================================================
    # Serialización de filtros activos
    # =========================================================================

    def active_dict(self) -> dict[str, Any]:
        """
        Retorna solo los filtros realmente activos.

        Útil para responder filters_applied al frontend sin llenar el JSON con
        campos null o flags false.
        """
        result: dict[str, Any] = {}

        for field_name in self.EXACT_FIELDS + self.RANGE_FIELDS:
            value = getattr(self, field_name)

            if value is None:
                continue

            if isinstance(value, list) and not value:
                continue

            result[field_name] = value

        for field_name in self.FLAG_FIELDS:
            if getattr(self, field_name):
                result[field_name] = True

        return result

    def is_empty(self) -> bool:
        """
        True si no hay ningún filtro activo.
        """
        return not bool(self.active_dict())

    # =========================================================================
    # Helpers internos
    # =========================================================================

    @staticmethod
    def _normalize_value(value: Any) -> Optional[str]:
        """
        Normaliza valores textuales para comparación robusta.

        Ejemplos:
            "Pocitos" -> "POCITOS"
            "Cordón"  -> "CORDON"
        """
        if value is None:
            return None

        normalized = normalize_for_match(str(value))

        return normalized or None

    @classmethod
    def _same_text(cls, left: Any, right: Any) -> bool:
        """
        Compara dos textos ignorando tildes, mayúsculas y espacios repetidos.
        """
        left_norm = cls._normalize_value(left)
        right_norm = cls._normalize_value(right)

        if left_norm is None or right_norm is None:
            return False

        return left_norm == right_norm

    @classmethod
    def _value_in_text_list(cls, value: Any, candidates: list[Any]) -> bool:
        """
        Verifica si value coincide con alguno de los candidatos normalizados.
        """
        value_norm = cls._normalize_value(value)

        if value_norm is None:
            return False

        candidate_norms = {
            normalized
            for normalized in (cls._normalize_value(candidate) for candidate in candidates)
            if normalized
        }

        return value_norm in candidate_norms

    @staticmethod
    def _as_list(value: Union[str, list[str]]) -> list[str]:
        if isinstance(value, list):
            return [str(item) for item in value if item is not None]

        return [str(value)]

    @staticmethod
    def _first_float(*values: Any) -> Optional[float]:
        """
        Retorna el primer valor convertible a float.
        """
        for value in values:
            converted = safe_float(value)
            if converted is not None:
                return converted

        return None

    @staticmethod
    def _missing_passes(strict_missing: bool) -> bool:
        """
        Define qué ocurre si un filtro activo no encuentra metadata.

        strict_missing=True:
            Si el usuario pide max_price y el listing no tiene price_fixed,
            el listing no pasa.

        strict_missing=False:
            Replica el comportamiento más laxo del flujo local original:
            si falta metadata, no se descarta el listing.
        """
        return not strict_missing

    # =========================================================================
    # Conversión a callable FAISS
    # =========================================================================

    def to_filter_fn(
        self,
        strict_missing: bool = True,
    ) -> Optional[Callable[[dict[str, Any]], bool]]:
        """
        Convierte los filtros en un callable compatible con FAISS/LangChain.

        La función recibe metadata de un documento y retorna True/False.

        Se usa un callable, en lugar de un dict, porque permite:

        - rangos numéricos;
        - filtro por múltiples barrios;
        - lógica especial para parking;
        - fallback barrio_fixed/barrio;
        - lógica AND entre filtros.

        Args:
            strict_missing:
                Si True, cuando un filtro activo depende de un campo ausente
                o nulo en metadata, el documento se descarta.

                Esto es preferible en GCP porque los filtros del frontend son
                filtros duros. Si se quiere replicar el comportamiento local
                más laxo, se puede llamar con strict_missing=False.

        Returns:
            Callable de filtrado, o None si no hay filtros activos.
        """
        if self.is_empty():
            return None

        filters = self

        def _filter(meta: dict[str, Any]) -> bool:
            metadata = meta or {}

            # -----------------------------------------------------------------
            # Segmentación
            # -----------------------------------------------------------------
            if filters.operation_type:
                if not self._same_text(
                    metadata.get("operation_type"),
                    filters.operation_type,
                ):
                    return False

            if filters.property_type:
                if not self._same_text(
                    metadata.get("property_type"),
                    filters.property_type,
                ):
                    return False

            if filters.barrio:
                barrios = self._as_list(filters.barrio)
                metadata_barrio = (
                    metadata.get("barrio_fixed")
                    or metadata.get("barrio")
                )

                if not self._value_in_text_list(metadata_barrio, barrios):
                    return False

            # -----------------------------------------------------------------
            # Precio
            # -----------------------------------------------------------------
            if filters.min_price is not None or filters.max_price is not None:
                price = safe_float(metadata.get("price_fixed"))

                if price is None:
                    return self._missing_passes(strict_missing)

                if filters.min_price is not None and price < filters.min_price:
                    return False

                if filters.max_price is not None and price > filters.max_price:
                    return False

            if filters.max_price_m2 is not None:
                price_m2 = safe_float(metadata.get("price_m2"))

                if price_m2 is None:
                    return self._missing_passes(strict_missing)

                if price_m2 > filters.max_price_m2:
                    return False

            # -----------------------------------------------------------------
            # Dormitorios
            # -----------------------------------------------------------------
            if filters.min_bedrooms is not None or filters.max_bedrooms is not None:
                bedrooms = safe_int(metadata.get("bedrooms"))

                if bedrooms is None:
                    return self._missing_passes(strict_missing)

                if filters.min_bedrooms is not None and bedrooms < filters.min_bedrooms:
                    return False

                if filters.max_bedrooms is not None and bedrooms > filters.max_bedrooms:
                    return False

            # -----------------------------------------------------------------
            # Superficie
            # -----------------------------------------------------------------
            if filters.min_surface is not None or filters.max_surface is not None:
                surface = self._first_float(
                    metadata.get("surface_covered"),
                    metadata.get("surface_total"),
                )

                if surface is None:
                    return self._missing_passes(strict_missing)

                if filters.min_surface is not None and surface < filters.min_surface:
                    return False

                if filters.max_surface is not None and surface > filters.max_surface:
                    return False

            # -----------------------------------------------------------------
            # Entorno urbano
            # -----------------------------------------------------------------
            if filters.max_dist_plaza is not None:
                dist_plaza = safe_float(metadata.get("dist_plaza"))

                if dist_plaza is None:
                    return self._missing_passes(strict_missing)

                if dist_plaza > filters.max_dist_plaza:
                    return False

            if filters.max_dist_playa is not None:
                dist_playa = safe_float(metadata.get("dist_playa"))

                if dist_playa is None:
                    return self._missing_passes(strict_missing)

                if dist_playa > filters.max_dist_playa:
                    return False

            # -----------------------------------------------------------------
            # Amenities requeridas
            # -----------------------------------------------------------------
            amenity_flags = [
                "has_pool",
                "has_gym",
                "has_elevator",
                "has_parrillero",
                "has_terrace",
                "has_rooftop",
                "has_security",
                "has_storage",
                "has_party_room",
                "has_green_area",
                "has_playground",
                "has_visitor_parking",
            ]

            for flag in amenity_flags:
                required = getattr(filters, flag)

                if required and not safe_bool(metadata.get(flag)):
                    return False

            # Parking se trata como caso especial:
            # en el índice puede no existir has_parking, pero sí garages.
            if filters.has_parking:
                has_parking_flag = safe_bool(metadata.get("has_parking"))
                garages = safe_float(metadata.get("garages"))

                if not has_parking_flag and not (garages is not None and garages > 0):
                    return False

            return True

        return _filter

    # =========================================================================
    # Compatibilidad / debug
    # =========================================================================

    def debug_summary(self) -> dict[str, Any]:
        """
        Resumen explícito para logs o pruebas.
        """
        return {
            "is_empty": self.is_empty(),
            "active_filters": self.active_dict(),
            "strict_default": True,
        }
