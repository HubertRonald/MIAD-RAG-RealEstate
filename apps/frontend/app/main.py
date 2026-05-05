from __future__ import annotations

import os
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from components.debug_panel import render_ask_context, render_debug_panel
from components.map_view import render_map
from components.property_cards import render_answer_block, render_listings_table, render_property_cards
from components.search_panel import render_ask_form, render_mode_carousel, render_recommend_form
from services.backend_client import BackendClient, BackendClientError

load_dotenv()


st.set_page_config(
    page_title="Su Casa Ya · Recomendador RAG",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
:root {
  --app-bg: #f8fbff;
  --surface: #ffffff;
  --surface-soft: #f1f7ff;
  --navy: #172033;
  --muted: #64748b;
  --blue: #2563eb;
  --blue-soft: #dbeafe;
  --orange: #f97316;
  --orange-soft: #fff3e7;
  --border: #dbe4f0;
}

.stApp {
  background:
    radial-gradient(circle at top left, rgba(37, 99, 235, 0.08), transparent 32rem),
    radial-gradient(circle at top right, rgba(249, 115, 22, 0.07), transparent 28rem),
    var(--app-bg);
  color: var(--navy);
}

section[data-testid="stSidebar"] {
  background: #ffffff;
  border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] * {
  color: var(--navy);
}

h1, h2, h3 {
  color: var(--navy);
  letter-spacing: -0.03em;
}

div[data-testid="stForm"],
div[data-testid="stExpander"],
div[data-testid="stVerticalBlockBorderWrapper"] {
  border-color: var(--border) !important;
  border-radius: 18px !important;
}

div[data-testid="stMetric"] {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 1rem;
}

.stButton > button {
  border-radius: 12px;
  border: 1px solid var(--border);
  background: #ffffff;
}

.stButton > button[kind="primary"],
button[kind="primary"] {
  background: linear-gradient(135deg, var(--orange), #fb923c);
  border: 0;
  color: white;
}

div[data-baseweb="tag"] {
  background-color: var(--orange) !important;
}

div[data-testid="stAlert"] {
  border-radius: 14px;
}

.property-card, .answer-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-left: 5px solid var(--orange);
  border-radius: 18px;
  padding: 1.25rem;
  box-shadow: 0 12px 35px rgba(15, 23, 42, 0.06);
}

.hero-card {
  background:
    linear-gradient(135deg, rgba(219, 234, 254, 0.88), rgba(255, 255, 255, 0.95)),
    linear-gradient(135deg, rgba(249, 115, 22, 0.08), transparent);
  border: 1px solid var(--border);
  border-radius: 24px;
  padding: 2rem;
  box-shadow: 0 18px 45px rgba(15, 23, 42, 0.08);
}
</style>
"""


def get_client() -> BackendClient | None:
    try:
        return BackendClient.from_env()
    except BackendClientError as exc:
        st.sidebar.error(str(exc))
        return None


def render_sidebar(client: BackendClient | None) -> str:
    st.sidebar.markdown("## 🏠 Su Casa Ya")
    st.sidebar.caption("Prototipo fachada · RAG inmobiliario")

    backend_url = os.getenv("BACKEND_URL", "")
    st.sidebar.text_input("BACKEND_URL", value=backend_url or "No configurado", disabled=True)
    st.sidebar.caption(f"Auth mode: `{os.getenv('BACKEND_AUTH_MODE', 'auto')}`")

    if st.sidebar.button("Probar /health", use_container_width=True, disabled=client is None):
        try:
            assert client is not None
            st.sidebar.success("Backend disponible")
            st.sidebar.json(client.health())
        except Exception as exc:  # noqa: BLE001
            st.sidebar.error(str(exc))

    st.sidebar.divider()
    page = st.sidebar.radio(
        "Vista",
        options=["Recomendador", "Preguntas de mercado"],
        index=0,
    )
    st.sidebar.divider()
    st.sidebar.caption("El frontend no replica lógica analítica: solo arma payloads, llama al backend y renderiza la respuesta.")
    return page


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero-card">
          <p style="color:#f97316; font-weight:800; letter-spacing:.08em; text-transform:uppercase; font-size:.78rem;">
            Montevideo · Recomendador Inmobil-IA-rio
          </p>
          <h1>Encuentra una vivienda que calce con tu vida, no solo con tus filtros.</h1>
          <p style="color:#475569; font-size:1rem; max-width:850px;">
            Prototipo para probar recomendaciones inmobiliarias con RAG: texto libre,
            filtros estructurados, mapa de propiedades y explicación generada por Gemini.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def run_recommendation_page(client: BackendClient | None) -> None:
    st.markdown("## Prueba el nuevo recomendador")
    mode_idx = render_mode_carousel()
    submitted, payload = render_recommend_form(mode_idx)

    if submitted:
        if client is None:
            st.error("Configura BACKEND_URL antes de consultar.")
            return
        try:
            with st.spinner("Consultando recomendaciones en el backend..."):
                response = client.recommend(payload)
            st.session_state["last_recommend_payload"] = payload
            st.session_state["last_recommend_response"] = response
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))

    response = st.session_state.get("last_recommend_response")
    last_payload = st.session_state.get("last_recommend_payload")

    if response:
        render_answer_block(response)

        render_map(response)

        listings_count = len(response.get("listings_used") or [])
        map_count = len(response.get("map_points") or [])
        response_time = response.get("response_time_sec", "—")

        st.markdown("### Indicadores")
        metric_cols = st.columns(3)
        metric_cols[0].metric("Propiedades", listings_count)
        metric_cols[1].metric("Puntos mapa", map_count)
        metric_cols[2].metric("Tiempo", response_time)

        if response.get("filters_applied"):
            with st.expander("Filtros aplicados", expanded=False):
                st.json(response.get("filters_applied"))

        render_property_cards(response.get("listings_used") or [])
        render_listings_table(response.get("listings_used") or [])
        render_debug_panel(last_payload, response)


def run_ask_page(client: BackendClient | None) -> None:
    st.markdown("## Preguntas de mercado")
    submitted, payload = render_ask_form()

    if submitted:
        if client is None:
            st.error("Configura BACKEND_URL antes de consultar.")
            return
        try:
            with st.spinner("Consultando contexto y generando respuesta..."):
                response = client.ask(payload)
            st.session_state["last_ask_payload"] = payload
            st.session_state["last_ask_response"] = response
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))

    response = st.session_state.get("last_ask_response")
    last_payload = st.session_state.get("last_ask_payload")

    if response:
        st.markdown("### Respuesta")
        st.markdown(f"<div class='answer-box'>{response.get('answer', 'Sin respuesta')}</div>", unsafe_allow_html=True)
        render_ask_context(response)
        render_debug_panel(last_payload, response)


def main() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    client = get_client()
    page = render_sidebar(client)
    render_hero()

    if page == "Recomendador":
        run_recommendation_page(client)
    else:
        run_ask_page(client)


if __name__ == "__main__":
    main()
