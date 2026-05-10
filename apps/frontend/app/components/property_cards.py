from __future__ import annotations

import base64
import html
from urllib.parse import urlparse
from typing import Any

import pandas as pd
import requests
import streamlit as st

from utils.formatting import (
    as_list,
    first_non_empty,
    format_distance,
    format_price,
    format_score,
    get_listing_barrio,
    get_listing_image,
    get_listing_title,
    title_case,
)


def format_card_number(value: Any, default: str = "—") -> str:
    """Formatea números de cards evitando decimales innecesarios.

    Ejemplos:
    2.0 -> 2
    140.0 -> 140
    82.5 -> 82.5
    None -> —
    """
    if value in (None, "", [], {}):
        return default

    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)

    if number.is_integer():
        return str(int(number))

    return f"{number:.1f}".rstrip("0").rstrip(".")


def format_card_area(value: Any, default: str = "—") -> str:
    number = format_card_number(value, default=default)

    if number == default:
        return default

    return f"{number} m²"


def _is_missing(value: Any) -> bool:
    if value is None:
        return True

    if isinstance(value, str) and not value.strip():
        return True

    if isinstance(value, (list, tuple, dict, set)) and len(value) == 0:
        return True

    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _is_zeroish(value: Any) -> bool:
    try:
        return float(value) == 0.0
    except (TypeError, ValueError):
        return False


INVALID_IMAGE_VALUES = {
    "",
    "0",
    "0.0",
    "none",
    "null",
    "nan",
    "na",
    "n/a",
    "-",
    "[]",
    "{}",
}


def normalize_image_url(value: Any) -> str | None:
    """Normaliza posibles campos de imagen.

    Evita intentar renderizar valores como 0, '0', None, NaN o strings vacíos.
    También acepta listas/dicts comunes de imágenes.
    """
    if _is_missing(value):
        return None

    if isinstance(value, dict):
        for key in ("url", "secure_url", "src", "href", "image_url", "thumbnail"):
            normalized = normalize_image_url(value.get(key))
            if normalized:
                return normalized
        return None

    if isinstance(value, (list, tuple, set)):
        for item in value:
            normalized = normalize_image_url(item)
            if normalized:
                return normalized
        return None

    raw = str(value).strip().strip('"').strip("'")

    if raw.lower() in INVALID_IMAGE_VALUES:
        return None

    if raw.startswith("//"):
        raw = f"https:{raw}"

    # Evita mixed-content cuando la app corre bajo HTTPS en run.app.
    if raw.startswith("http://"):
        raw = f"https://{raw.removeprefix('http://')}"

    parsed = urlparse(raw)

    if parsed.scheme not in {"http", "https"}:
        return None

    if not parsed.netloc:
        return None

    return raw


def resolve_listing_image_url(listing: dict[str, Any]) -> str | None:
    """Busca una imagen válida en varios campos posibles del listing."""
    candidates = [
        get_listing_image(listing),
        listing.get("image_url"),
        listing.get("thumbnail_url"),
        listing.get("thumbnail"),
        listing.get("picture_url"),
        listing.get("cover_image_url"),
        listing.get("main_image_url"),
        listing.get("image"),
        listing.get("images"),
        listing.get("photos"),
        listing.get("pictures"),
    ]

    for candidate in candidates:
        normalized = normalize_image_url(candidate)
        if normalized:
            return normalized

    return None


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def fetch_image_as_data_uri(image_url: str) -> str | None:
    """Descarga una imagen remota desde el servidor y la devuelve como data URI.

    Esto evita depender de que el navegador cargue directamente imágenes externas
    desde dominios como http2.mlstatic.com cuando la app está detrás de Cloud Run/IAP.
    """
    if not image_url:
        return None

    try:
        response = requests.get(
            image_url,
            timeout=8,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0 Safari/537.36"
                ),
                "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
            },
        )
        response.raise_for_status()

        content_type = response.headers.get("content-type", "").split(";")[0].strip()

        if not content_type.startswith("image/"):
            return None

        encoded = base64.b64encode(response.content).decode("utf-8")
        return f"data:{content_type};base64,{encoded}"

    except requests.RequestException:
        return None


def render_listing_image(image_url: str | None, title: str) -> None:
    if not image_url:
        st.markdown(
            """
            <div class="image-placeholder">
                <div class="image-placeholder-icon">🏠</div>
                <div>Sin imagen</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    data_uri = fetch_image_as_data_uri(image_url)

    if not data_uri:
        st.markdown(
            """
            <div class="image-placeholder">
                <div class="image-placeholder-icon">🏠</div>
                <div>Imagen no disponible</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    safe_src = html.escape(data_uri, quote=True)
    safe_alt = html.escape(title or "Imagen de propiedad", quote=True)

    st.markdown(
        f"""
        <div class="listing-image-frame">
            <img
                src="{safe_src}"
                alt="{safe_alt}"
                loading="lazy"
            />
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_listing_image(image_url: str | None, title: str) -> None:
    if not image_url:
        st.markdown(
            """
            <div class="image-placeholder">
                <div class="image-placeholder-icon">🏠</div>
                <div>Sin imagen</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    safe_url = html.escape(image_url, quote=True)
    safe_alt = html.escape(title or "Imagen de propiedad", quote=True)

    st.markdown(
        f"""
        <div class="listing-image-frame">
            <img
                src="{safe_url}"
                alt="{safe_alt}"
                referrerpolicy="no-referrer"
                loading="lazy"
            />
        </div>
        """,
        unsafe_allow_html=True,
    )


def resolve_parking_or_floor_stat(listing: dict[str, Any]) -> tuple[str, str]:
    """Define la cuarta caja de la card.

    Regla:
    - Si llega cochera/garages, mostrar Cochera, incluso si es 0.
    - Si no llega cochera pero llega piso válido distinto de 0, mostrar Piso.
    - Si no llega ninguno, mostrar Cochera 0.
    """
    garages_raw = listing.get("garages")
    floor_raw = listing.get("floor")

    if not _is_missing(garages_raw):
        return "Cochera", format_card_number(garages_raw)

    if not _is_missing(floor_raw) and not _is_zeroish(floor_raw):
        return "Piso", format_card_number(floor_raw)

    return "Cochera", "0"


def render_stat_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">{label}</div>
            <div class="stat-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    

def render_answer_block(
    response: dict[str, Any],
    title: str | None = "Resumen recomendado",
) -> None:
    answer = response.get("answer")
    if not answer:
        return

    if title:
        st.markdown(f"### {title}")

    st.markdown(
        f"""
        <div class="answer-card">
            {answer}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_property_cards(
    listings: list[dict[str, Any]],
    initial_visible: int = 3,
) -> None:
    if not listings:
        st.info("No llegaron propiedades recomendadas para mostrar.")
        return

    st.markdown("### Propiedades recomendadas")

    show_all = bool(st.session_state.get("show_all_listings", False))
    visible_listings = listings if show_all else listings[:initial_visible]

    for idx, listing in enumerate(visible_listings, start=1):
        rank = listing.get("rank") or idx
        title = get_listing_title(listing)
        barrio = get_listing_barrio(listing)
        image_url = resolve_listing_image_url(listing)

        price = format_price(
            listing.get("price_fixed"),
            listing.get("currency_fixed"),
        )
        operation = title_case(listing.get("operation_type"))
        prop_type = title_case(listing.get("property_type"))

        bedrooms = format_card_number(
            first_non_empty(listing.get("bedrooms"), default=None)
        )
        
        bathrooms = format_card_number(
            first_non_empty(listing.get("bathrooms"), default=None)
        )
        
        surface = format_card_area(
            first_non_empty(
                listing.get("surface_total"),
                listing.get("surface_covered"),
                default=None,
            )
        )
        
        last_stat_label, last_stat_value = resolve_parking_or_floor_stat(listing)

        match_score = format_score(
            first_non_empty(
                listing.get("match_score"),
                listing.get("semantic_score"),
                default=None,
            )
        )

        url = listing.get("url")
        description = first_non_empty(
            listing.get("description_clean"),
            listing.get("description"),
            listing.get("retrieval_snippet"),
            default="",
        )
        amenities = as_list(listing.get("amenities"))[:5]

        with st.container(border=True):
            left_col, right_col = st.columns([1.05, 2.35], gap="large")

            with left_col:
                st.markdown(
                    f"""
                    <div class="rank-pill">#{rank}</div>
                    """,
                    unsafe_allow_html=True,
                )

                render_listing_image(image_url, title)

            with right_col:
                st.markdown(
                   f"""
                   <div class="property-header-row">
                       <div class="property-header-main">
                           <div class="property-meta">
                               {barrio} · {operation} · {prop_type}
                           </div>
                           <div class="property-title">{title}</div>
                           <div class="property-price">{price}</div>
                       </div>
                       <div class="score-badge">{match_score} ajuste</div>
                   </div>
                   """,
                   unsafe_allow_html=True,
                )

                metric_cols = st.columns(4)
                
                with metric_cols[0]:
                    render_stat_card("Dorm.", bedrooms)
                
                with metric_cols[1]:
                    render_stat_card("Baños", bathrooms)
                
                with metric_cols[2]:
                    render_stat_card("Sup.", surface)
                
                with metric_cols[3]:
                    render_stat_card(last_stat_label, last_stat_value)

                env_cols = st.columns(3)
                env_cols[0].caption(
                    f"🔵 Playa: {format_distance(listing.get('dist_playa'))}"
                )
                env_cols[1].caption(
                    f"🟢 Plaza: {format_distance(listing.get('dist_plaza'))}"
                )
                env_cols[2].caption(
                    f"🟠 Escuelas 800m: "
                    f"{first_non_empty(listing.get('n_escuelas_800m'), default='—')}"
                )

                if description:
                    st.caption(
                        str(description)[:340]
                        + ("..." if len(str(description)) > 340 else "")
                    )

                if amenities:
                    chips = " ".join(
                        f"<span class='soft-chip'>{amenity}</span>"
                        for amenity in amenities
                    )
                    st.markdown(chips, unsafe_allow_html=True)

                if url:
                    st.link_button("Ver publicación", str(url), use_container_width=False)

    remaining = len(listings) - initial_visible
    if remaining > 0 and not show_all:
        if st.button(
            f"Ver {remaining} propiedades más ↗",
            key="show_more_listings",
            use_container_width=False,
        ):
            st.session_state["show_all_listings"] = True
            st.rerun()


def render_listings_table(listings: list[dict[str, Any]]) -> None:
    if not listings:
        return

    preferred_cols = [
        "rank",
        "title",
        "barrio_fixed",
        "operation_type",
        "property_type",
        "price_fixed",
        "currency_fixed",
        "bedrooms",
        "bathrooms",
        "surface_total",
        "surface_covered",
        "dist_playa",
        "dist_plaza",
        "match_score",
        "semantic_score",
        "rerank_score",
        "url",
    ]

    df = pd.DataFrame(listings)
    visible_cols = [col for col in preferred_cols if col in df.columns]

    if visible_cols:
        df = df[visible_cols]

    with st.expander("Ver tabla técnica de propiedades usadas por la respuesta", expanded=False):
        st.dataframe(df, use_container_width=True, hide_index=True)
