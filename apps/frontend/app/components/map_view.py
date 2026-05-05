from __future__ import annotations

from typing import Any

import pandas as pd
import pydeck as pdk
import streamlit as st

from utils.formatting import MONTEVIDEO_CENTER, derive_map_points, map_points_to_dataframe


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _compute_zoom(df: pd.DataFrame) -> float:
    """Ajuste simple de zoom según dispersión de puntos."""
    if df.empty or "lat" not in df or "lon" not in df:
        return 11.5

    lat_span = float(df["lat"].max() - df["lat"].min())
    lon_span = float(df["lon"].max() - df["lon"].min())
    span = max(lat_span, lon_span)

    if span <= 0.003:
        return 14.5
    if span <= 0.008:
        return 13.5
    if span <= 0.02:
        return 12.5
    if span <= 0.05:
        return 11.5
    return 10.5


def render_map(response: dict[str, Any]) -> None:
    points = derive_map_points(response)

    if not points:
        st.info("No hay puntos de mapa disponibles. El backend no envió `map_points` ni listings con lat/lon.")
        return

    df = map_points_to_dataframe(points)

    if df.empty:
        st.info("Los puntos recibidos no tienen coordenadas válidas.")
        return

    df["lat"] = df["lat"].apply(_safe_float)
    df["lon"] = df["lon"].apply(_safe_float)
    df = df.dropna(subset=["lat", "lon"])

    if df.empty:
        st.info("Los puntos recibidos no tienen coordenadas válidas después de normalizarlas.")
        return

    st.markdown("### Mapa de recomendaciones")

    center_lat = float(df["lat"].mean()) if "lat" in df else MONTEVIDEO_CENTER["lat"]
    center_lon = float(df["lon"].mean()) if "lon" in df else MONTEVIDEO_CENTER["lon"]
    zoom = _compute_zoom(df)

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position="[lon, lat]",
        get_radius=90,
        get_fill_color="[249, 115, 22, 190]",
        get_line_color="[30, 64, 175, 180]",
        line_width_min_pixels=1,
        pickable=True,
    )

    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=zoom,
        pitch=0,
    )

    tooltip = {
        "html": """
        <div style="font-family:Inter, sans-serif; padding:6px;">
          <b>#{rank} {label}</b><br/>
          {barrio}<br/>
          {precio}
        </div>
        """,
        "style": {
            "backgroundColor": "white",
            "color": "#172033",
            "border": "1px solid #e2e8f0",
            "borderRadius": "10px",
        },
    }

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style="light",
    )

    st.pydeck_chart(deck, use_container_width=True, height=520)

    with st.expander("Puntos usados en el mapa", expanded=False):
        st.dataframe(df, use_container_width=True, hide_index=True)
