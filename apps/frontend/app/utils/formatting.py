from __future__ import annotations

import json
from typing import Any

import pandas as pd


MONTEVIDEO_CENTER = {"lat": -34.9011, "lon": -56.1645}


def as_list(value: Any) -> list[Any]:
    """Return value as list, handling JSON-encoded lists used by some BigQuery exports."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        if value.startswith("["):
            try:
                parsed = json.loads(value)
                return parsed if isinstance(parsed, list) else [parsed]
            except json.JSONDecodeError:
                return [value]
        return [value]
    return [value]


def first_non_empty(*values: Any, default: str = "") -> Any:
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return default


def title_case(value: Any) -> str:
    if value is None:
        return "—"
    text = str(value).replace("_", " ").strip()
    if not text:
        return "—"
    return text.title()


def normalize_barrio(value: str) -> str:
    return value.strip().upper()


def format_price(price: Any, currency: Any = None) -> str:
    if price in (None, "", "nan"):
        return "Precio no disponible"
    currency_text = str(currency or "USD").upper()
    try:
        amount = float(price)
        if amount.is_integer():
            amount_text = f"{int(amount):,}".replace(",", ".")
        else:
            amount_text = f"{amount:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")
        return f"{currency_text} {amount_text}"
    except (TypeError, ValueError):
        return f"{currency_text} {price}"


def format_score(value: Any) -> str:
    if value in (None, "", "nan"):
        return "—"
    try:
        number = float(value)
        if 0 <= number <= 1:
            return f"{number * 100:.0f}%"
        return f"{number:.2f}"
    except (TypeError, ValueError):
        return str(value)


def format_distance(value: Any) -> str:
    if value in (None, "", "nan"):
        return "—"
    try:
        meters = float(value)
        if meters >= 1000:
            return f"{meters / 1000:.1f} km"
        return f"{meters:.0f} m"
    except (TypeError, ValueError):
        return str(value)


def get_listing_image(listing: dict[str, Any]) -> str | None:
    thumbnail = listing.get("thumbnail_url")
    if thumbnail:
        return str(thumbnail)

    images = as_list(listing.get("image_urls"))
    for image_url in images:
        if image_url:
            return str(image_url)
    return None


def get_listing_title(listing: dict[str, Any]) -> str:
    return str(first_non_empty(listing.get("title_clean"), listing.get("title"), listing.get("label"), default="Propiedad recomendada"))


def get_listing_barrio(listing: dict[str, Any]) -> str:
    return str(first_non_empty(listing.get("barrio_fixed"), listing.get("barrio"), default="Montevideo"))


def _safe_float(value: Any) -> float | None:
    if value in (None, "", "nan", "NaN", "null", "None"):
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if pd.isna(number):
        return None

    return number


def _extract_lat_lon(item: dict[str, Any]) -> tuple[float, float] | None:
    """Extrae coordenadas aceptando distintos nombres posibles."""
    lat = first_non_empty(
        item.get("lat"),
        item.get("latitude"),
        item.get("geo_lat"),
        item.get("location_lat"),
        item.get("latitud"),
        default=None,
    )

    lon = first_non_empty(
        item.get("lon"),
        item.get("lng"),
        item.get("longitude"),
        item.get("geo_lon"),
        item.get("geo_lng"),
        item.get("location_lon"),
        item.get("location_lng"),
        item.get("longitud"),
        default=None,
    )

    lat_f = _safe_float(lat)
    lon_f = _safe_float(lon)

    if lat_f is None or lon_f is None:
        return None

    if not (-90 <= lat_f <= 90 and -180 <= lon_f <= 180):
        return None

    return lat_f, lon_f


def _safe_rank(value: Any, fallback: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return fallback


def _map_point_key(point: dict[str, Any]) -> str:
    point_id = first_non_empty(point.get("id"), point.get("listing_id"), default=None)
    if point_id:
        return f"id:{point_id}"

    rank = first_non_empty(point.get("rank"), default=None)
    if rank:
        return f"rank:{rank}"

    coords = _extract_lat_lon(point)
    if coords:
        lat, lon = coords
        return f"coord:{lat:.6f}:{lon:.6f}"

    return f"unknown:{id(point)}"


def _map_point_from_listing(
    listing: dict[str, Any],
    idx: int,
) -> dict[str, Any] | None:
    coords = _extract_lat_lon(listing)

    if coords is None:
        return None

    lat, lon = coords
    rank = _safe_rank(listing.get("rank"), idx)

    return {
        "id": first_non_empty(listing.get("id"), listing.get("listing_id"), default=None),
        "lat": lat,
        "lon": lon,
        "label": get_listing_title(listing),
        "barrio": get_listing_barrio(listing),
        "price_fixed": listing.get("price_fixed"),
        "currency_fixed": listing.get("currency_fixed"),
        "match_score": listing.get("match_score"),
        "rank": rank,
    }


def _map_point_from_backend(
    point: dict[str, Any],
    idx: int,
    listings_by_id: dict[str, dict[str, Any]],
    listings_by_rank: dict[int, dict[str, Any]],
) -> dict[str, Any] | None:
    coords = _extract_lat_lon(point)

    if coords is None:
        return None

    lat, lon = coords
    rank = _safe_rank(point.get("rank"), idx)

    point_id = first_non_empty(
        point.get("id"),
        point.get("listing_id"),
        default=None,
    )

    source_listing = None

    if point_id:
        source_listing = listings_by_id.get(str(point_id))

    if source_listing is None:
        source_listing = listings_by_rank.get(rank)

    return {
        "id": first_non_empty(
            point.get("id"),
            point.get("listing_id"),
            source_listing.get("id") if source_listing else None,
            source_listing.get("listing_id") if source_listing else None,
            default=None,
        ),
        "lat": lat,
        "lon": lon,
        "label": first_non_empty(
            point.get("label"),
            point.get("title"),
            get_listing_title(source_listing) if source_listing else None,
            default="Propiedad",
        ),
        "barrio": first_non_empty(
            point.get("barrio"),
            point.get("barrio_fixed"),
            get_listing_barrio(source_listing) if source_listing else None,
            default="—",
        ),
        "price_fixed": first_non_empty(
            point.get("price_fixed"),
            source_listing.get("price_fixed") if source_listing else None,
            default=None,
        ),
        "currency_fixed": first_non_empty(
            point.get("currency_fixed"),
            source_listing.get("currency_fixed") if source_listing else None,
            default=None,
        ),
        "match_score": first_non_empty(
            point.get("match_score"),
            source_listing.get("match_score") if source_listing else None,
            default=None,
        ),
        "rank": rank,
    }


def derive_map_points(response: dict[str, Any]) -> list[dict[str, Any]]:
    """Construye puntos de mapa de forma tolerante.

    Regla:
    - Usar `map_points` del backend cuando existan.
    - Completar puntos faltantes desde `listings_used` si tienen coordenadas.
    - Ordenar por rank para que el mapa y las cards queden alineados.
    """
    listings = response.get("listings_used") or []

    listings_by_id: dict[str, dict[str, Any]] = {}
    listings_by_rank: dict[int, dict[str, Any]] = {}

    derived_from_listings: list[dict[str, Any]] = []

    for idx, listing in enumerate(listings, start=1):
        rank = _safe_rank(listing.get("rank"), idx)
        listings_by_rank[rank] = listing

        listing_id = first_non_empty(
            listing.get("id"),
            listing.get("listing_id"),
            default=None,
        )

        if listing_id:
            listings_by_id[str(listing_id)] = listing

        derived_point = _map_point_from_listing(listing, idx)
        if derived_point:
            derived_from_listings.append(derived_point)

    merged_points: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    backend_map_points = response.get("map_points") or []

    for idx, point in enumerate(backend_map_points, start=1):
        normalized_point = _map_point_from_backend(
            point=point,
            idx=idx,
            listings_by_id=listings_by_id,
            listings_by_rank=listings_by_rank,
        )

        if not normalized_point:
            continue

        key = _map_point_key(normalized_point)
        seen_keys.add(key)
        merged_points.append(normalized_point)

    # Completar con listings que no llegaron en map_points.
    for point in derived_from_listings:
        key = _map_point_key(point)

        if key in seen_keys:
            continue

        seen_keys.add(key)
        merged_points.append(point)

    return sorted(
        merged_points,
        key=lambda item: _safe_rank(item.get("rank"), 9999),
    )


def map_points_to_dataframe(points: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for idx, point in enumerate(points, start=1):
        coords = _extract_lat_lon(point)

        if coords is None:
            continue

        lat, lon = coords
        rank = _safe_rank(point.get("rank"), idx)

        rows.append(
            {
                "lat": lat,
                "lon": lon,
                "rank": rank,
                "label": point.get("label") or point.get("id") or "Propiedad",
                "barrio": point.get("barrio") or "—",
                "precio": format_price(
                    point.get("price_fixed"),
                    point.get("currency_fixed"),
                ),
                "match_score": point.get("match_score"),
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    return df.sort_values("rank").reset_index(drop=True)
