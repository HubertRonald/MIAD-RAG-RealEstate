from __future__ import annotations

from typing import Any

import pandas as pd
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
        image_url = get_listing_image(listing)

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

                if image_url:
                    st.image(image_url, use_container_width=True)
                else:
                    st.markdown(
                        """
                        <div class="image-placeholder">
                            <div class="image-placeholder-icon">🏠</div>
                            <div>Sin imagen</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

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
