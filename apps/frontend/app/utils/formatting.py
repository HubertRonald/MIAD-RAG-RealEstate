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


def derive_map_points(response: dict[str, Any]) -> list[dict[str, Any]]:
    """Prefer backend map_points; otherwise derive from listings_used lat/lon."""
    map_points = response.get("map_points") or []
    if map_points:
        return [point for point in map_points if point.get("lat") is not None and point.get("lon") is not None]

    derived: list[dict[str, Any]] = []
    for idx, listing in enumerate(response.get("listings_used") or [], start=1):
        lat = listing.get("lat")
        lon = listing.get("lon")
        if lat is None or lon is None:
            continue
        derived.append(
            {
                "id": listing.get("id"),
                "lat": lat,
                "lon": lon,
                "label": get_listing_title(listing),
                "barrio": get_listing_barrio(listing),
                "price_fixed": listing.get("price_fixed"),
                "currency_fixed": listing.get("currency_fixed"),
                "match_score": listing.get("match_score"),
                "rank": listing.get("rank") or idx,
            }
        )
    return derived


def map_points_to_dataframe(points: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for point in points:
        try:
            lat = float(point.get("lat"))
            lon = float(point.get("lon"))
        except (TypeError, ValueError):
            continue

        rows.append(
            {
                "lat": lat,
                "lon": lon,
                "rank": point.get("rank"),
                "label": point.get("label") or point.get("id") or "Propiedad",
                "barrio": point.get("barrio") or "—",
                "precio": format_price(point.get("price_fixed"), point.get("currency_fixed")),
                "match_score": point.get("match_score"),
            }
        )
    return pd.DataFrame(rows)
