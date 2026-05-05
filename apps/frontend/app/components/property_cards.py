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


def render_answer_block(response: dict[str, Any]) -> None:
    answer = response.get("answer")
    if not answer:
        return

    st.markdown("### Resumen recomendado")
    st.markdown(f"<div class='answer-box'>{answer}</div>", unsafe_allow_html=True)


def render_property_cards(listings: list[dict[str, Any]]) -> None:
    if not listings:
        st.info("No llegaron propiedades recomendadas en `listings_used`.")
        return

    st.markdown("### Propiedades recomendadas")

    for idx, listing in enumerate(listings, start=1):
        rank = listing.get("rank") or idx
        title = get_listing_title(listing)
        barrio = get_listing_barrio(listing)
        image_url = get_listing_image(listing)
        price = format_price(listing.get("price_fixed"), listing.get("currency_fixed"))
        operation = title_case(listing.get("operation_type"))
        prop_type = title_case(listing.get("property_type"))
        bedrooms = first_non_empty(listing.get("bedrooms"), default="—")
        bathrooms = first_non_empty(listing.get("bathrooms"), default="—")
        surface = first_non_empty(listing.get("surface_total"), listing.get("surface_covered"), default="—")
        match_score = format_score(first_non_empty(listing.get("match_score"), listing.get("semantic_score"), default=None))
        url = listing.get("url")
        description = first_non_empty(listing.get("description_clean"), listing.get("description"), listing.get("retrieval_snippet"), default="")
        amenities = as_list(listing.get("amenities"))[:5]

        with st.container(border=True):
            left_col, right_col = st.columns([1.1, 2.2], gap="large")
            with left_col:
                if image_url:
                    st.image(image_url, use_column_width=True)
                else:
                    st.markdown("<div class='image-placeholder'>Sin imagen</div>", unsafe_allow_html=True)

            with right_col:
                st.markdown(
                    f"""
                    <div class="card-topline">
                      <span class="rank-pill">#{rank}</span>
                      <span>{barrio}</span>
                      <span>·</span>
                      <span>{operation}</span>
                      <span>·</span>
                      <span>{prop_type}</span>
                    </div>
                    <h4 class="property-title">{title}</h4>
                    <div class="price-text">{price}</div>
                    """,
                    unsafe_allow_html=True,
                )

                metric_cols = st.columns(4)
                metric_cols[0].metric("Dorm.", bedrooms)
                metric_cols[1].metric("Baños", bathrooms)
                metric_cols[2].metric("Superficie", f"{surface} m²" if surface != "—" else "—")
                metric_cols[3].metric("Ajuste", match_score)

                env_cols = st.columns(3)
                env_cols[0].caption(f"🏖️ Playa: {format_distance(listing.get('dist_playa'))}")
                env_cols[1].caption(f"🌳 Plaza: {format_distance(listing.get('dist_plaza'))}")
                env_cols[2].caption(f"🏫 Escuelas 800m: {first_non_empty(listing.get('n_escuelas_800m'), default='—')}")

                if description:
                    st.caption(str(description)[:340] + ("..." if len(str(description)) > 340 else ""))

                if amenities:
                    chips = " ".join(f"<span class='chip'>{amenity}</span>" for amenity in amenities)
                    st.markdown(chips, unsafe_allow_html=True)

                if url:
                    st.link_button("Ver publicación", str(url), use_container_width=False)


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

    with st.expander("Tabla de propiedades usadas por la respuesta", expanded=False):
        st.dataframe(df, use_container_width=True, hide_index=True)
