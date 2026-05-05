from __future__ import annotations

from typing import Any

import pandas as pd
import pydeck as pdk
import streamlit as st

from utils.formatting import MONTEVIDEO_CENTER, derive_map_points, map_points_to_dataframe


def render_map(response: dict[str, Any]) -> None:
    points = derive_map_points(response)
    if not points:
        st.info("No hay puntos de mapa disponibles. El backend no envió `map_points` ni listings con lat/lon.")
        return

    df = map_points_to_dataframe(points)
    if df.empty:
        st.info("Los puntos recibidos no tienen coordenadas válidas.")
        return

    st.markdown("### Mapa de recomendaciones")

    center_lat = float(df["lat"].mean()) if "lat" in df else MONTEVIDEO_CENTER["lat"]
    center_lon = float(df["lon"].mean()) if "lon" in df else MONTEVIDEO_CENTER["lon"]

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position="[lon, lat]",
        get_radius=120,
        get_fill_color="[249, 115, 22, 180]",
        pickable=True,
    )

    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=11,
        pitch=0,
    )

    tooltip = {
        "html": "<b>#{rank} {label}</b><br/>{barrio}<br/>{precio}",
        "style": {"backgroundColor": "white", "color": "#172033"},
    }

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style="light",
    )
    st.pydeck_chart(deck, use_container_width=True)

    with st.expander("Puntos usados en el mapa", expanded=False):
        st.dataframe(df, use_container_width=True, hide_index=True)
