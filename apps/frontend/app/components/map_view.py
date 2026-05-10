from __future__ import annotations

from typing import Any

import math
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


def _add_display_offsets_for_overlapping_points(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega offsets mínimos para visualizar puntos con la misma coordenada.

    No modifica lat/lon originales. Solo crea plot_lat/plot_lon para renderizar.
    """
    if df.empty or "lat" not in df or "lon" not in df:
        return df

    df = df.copy()
    df["plot_lat"] = df["lat"]
    df["plot_lon"] = df["lon"]

    # Redondeamos para detectar puntos prácticamente iguales.
    df["_coord_key"] = (
        df["lat"].round(5).astype(str) + ":" + df["lon"].round(5).astype(str)
    )

    grouped = df.groupby("_coord_key", sort=False).groups

    for _, indexes in grouped.items():
        indexes = list(indexes)

        if len(indexes) <= 1:
            continue

        # Offset visual aproximado de 20-30 metros.
        radius = 0.00025

        for pos, row_idx in enumerate(indexes):
            angle = 2 * math.pi * pos / len(indexes)

            df.at[row_idx, "plot_lat"] = df.at[row_idx, "lat"] + radius * math.sin(angle)
            df.at[row_idx, "plot_lon"] = df.at[row_idx, "lon"] + radius * math.cos(angle)

    return df.drop(columns=["_coord_key"], errors="ignore")


def _format_tooltip_score(value: Any) -> str:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return "—"

    if score <= 1:
        score *= 100

    return f"{round(score)}%"


def render_map(
    response: dict[str, Any],
    *,
    show_title: bool = False,
    height: int = 340,
    show_points_debug: bool = False,
) -> None:
    points = derive_map_points(response)

    if not points:
        st.info(
            "No hay puntos de mapa disponibles. "
            "El backend no envió `map_points` ni listings con lat/lon."
        )
        return

    df = map_points_to_dataframe(points)

    if df.empty:
        st.info("Los puntos recibidos no tienen coordenadas válidas.")
        return

    df["lat"] = df["lat"].apply(_safe_float)
    df["lon"] = df["lon"].apply(_safe_float)
    df = df.dropna(subset=["lat", "lon"])
    df = _add_display_offsets_for_overlapping_points(df)
    df["match_score_text"] = df["match_score"].apply(_format_tooltip_score)

    if df.empty:
        st.info("Los puntos recibidos no tienen coordenadas válidas después de normalizarlas.")
        return

    if show_title:
        st.markdown("### Mapa de recomendaciones")

    center_lat = float(df["lat"].mean()) if "lat" in df else MONTEVIDEO_CENTER["lat"]
    center_lon = float(df["lon"].mean()) if "lon" in df else MONTEVIDEO_CENTER["lon"]
    zoom = _compute_zoom(df)

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position="[plot_lon, plot_lat]",
        get_radius=80,
        get_fill_color="[249, 115, 22, 210]",
        get_line_color="[37, 99, 235, 190]",
        line_width_min_pixels=1,
        pickable=True,
    )

    text_layer = pdk.Layer(
        "TextLayer",
        data=df,
        get_position="[plot_lon, plot_lat]",
        get_text="rank",
        get_size=13,
        get_color="[23, 32, 51, 255]",
        get_angle=0,
        get_text_anchor="middle",
        get_alignment_baseline="center",
        pickable=False,
    )

    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=zoom,
        pitch=0,
    )

    tooltip = {
        "html": """
        <div style="
            font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            min-width: 230px;
            max-width: 290px;
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.22);
            border: 1px solid rgba(226, 232, 240, 0.95);
            background: #ffffff;
        ">
          <div style="
              background: linear-gradient(135deg, #172033 0%, #1d4ed8 100%);
              color: #ffffff;
              padding: 11px 13px 10px 13px;
          ">
            <div style="
                font-size: 11px;
                font-weight: 800;
                letter-spacing: .08em;
                text-transform: uppercase;
                opacity: .82;
                margin-bottom: 4px;
            ">
              Recomendación #{rank}
            </div>
    
            <div style="
                font-size: 14px;
                line-height: 1.25;
                font-weight: 850;
            ">
              {label}
            </div>
          </div>
    
          <div style="
              padding: 12px 13px 13px 13px;
              color: #172033;
              background:
                radial-gradient(circle at top right, rgba(249, 115, 22, .10), transparent 90px),
                #ffffff;
          ">
            <div style="
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 10px;
                margin-bottom: 9px;
            ">
              <div style="
                  display: inline-flex;
                  align-items: center;
                  gap: 6px;
                  color: #475569;
                  font-size: 12px;
                  font-weight: 700;
              ">
                <span style="
                    width: 9px;
                    height: 9px;
                    border-radius: 999px;
                    background: #f97316;
                    display: inline-block;
                "></span>
                {barrio}
              </div>
    
              <div style="
                  background: #ecfccb;
                  color: #3f6212;
                  border-radius: 999px;
                  padding: 3px 8px;
                  font-size: 11px;
                  font-weight: 850;
                  white-space: nowrap;
              ">
                {match_score_text} ajuste
              </div>
            </div>
    
            <div style="
                display: flex;
                align-items: baseline;
                justify-content: space-between;
                gap: 10px;
                padding-top: 8px;
                border-top: 1px solid #e2e8f0;
            ">
              <span style="
                  color: #64748b;
                  font-size: 12px;
                  font-weight: 700;
              ">
                Precio
              </span>
    
              <span style="
                  color: #0f172a;
                  font-size: 18px;
                  font-weight: 900;
                  letter-spacing: -0.02em;
              ">
                {precio}
              </span>
            </div>
          </div>
        </div>
        """,
        "style": {
            "backgroundColor": "transparent",
            "border": "none",
            "padding": "0",
            "boxShadow": "none",
        },
    }

    deck = pdk.Deck(
        layers=[layer, text_layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style="light",
    )

    st.pydeck_chart(deck, use_container_width=True, height=height)

    if show_points_debug:
        with st.expander("Puntos usados en el mapa", expanded=False):
            st.dataframe(df, use_container_width=True, hide_index=True)
