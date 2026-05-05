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
        --brand-orange: #f97316;
        --brand-orange-soft: #fff3e7;
        --brand-blue: #2563eb;
        --ink: #172033;
        --muted: #64748b;
        --card: #ffffff;
        --line: #e5e7eb;
        --sand: #fffaf3;
    }

    .block-container {
        padding-top: 1.7rem;
        padding-bottom: 3rem;
    }

    .hero {
        background: linear-gradient(135deg, #fff7ed 0%, #ffffff 48%, #eff6ff 100%);
        border: 1px solid var(--line);
        border-radius: 28px;
        padding: 2rem 2.2rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.06);
    }

    .hero-kicker {
        color: var(--brand-orange);
        font-weight: 800;
        letter-spacing: .08em;
        text-transform: uppercase;
        font-size: .78rem;
        margin-bottom: .35rem;
    }

    .hero h1 {
        color: var(--ink);
        font-size: clamp(2rem, 3vw, 3rem);
        line-height: 1.02;
        margin: 0 0 .55rem 0;
    }

    .hero p {
        color: var(--muted);
        max-width: 850px;
        font-size: 1.02rem;
        margin: 0;
    }

    .mode-carousel {
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 24px;
        padding: 1.25rem 1.4rem;
        margin: .6rem 0 .85rem 0;
        box-shadow: 0 12px 28px rgba(15, 23, 42, 0.05);
    }

    .mode-eyebrow {
        color: var(--brand-orange);
        font-weight: 800;
        font-size: .75rem;
        text-transform: uppercase;
        letter-spacing: .08em;
    }

    .mode-carousel h3 {
        margin: .25rem 0 .35rem 0;
        color: var(--ink);
    }

    .mode-carousel p {
        margin: 0 0 .75rem 0;
        color: var(--muted);
    }

    .mode-example {
        background: var(--brand-orange-soft);
        border: 1px solid #fed7aa;
        border-radius: 16px;
        padding: .75rem .9rem;
        color: #7c2d12;
        font-size: .92rem;
    }

    .answer-box {
        background: #ffffff;
        border-left: 6px solid var(--brand-orange);
        border-radius: 18px;
        padding: 1.1rem 1.25rem;
        box-shadow: 0 10px 26px rgba(15, 23, 42, 0.06);
        color: var(--ink);
        margin-bottom: 1rem;
    }

    .card-topline {
        color: var(--muted);
        display: flex;
        align-items: center;
        gap: .4rem;
        flex-wrap: wrap;
        font-size: .88rem;
        margin-bottom: .2rem;
    }

    .rank-pill {
        background: var(--brand-orange);
        color: white;
        border-radius: 999px;
        padding: .15rem .55rem;
        font-weight: 800;
    }

    .property-title {
        margin: .15rem 0 .2rem 0 !important;
        color: var(--ink);
        font-size: 1.18rem !important;
    }

    .price-text {
        font-size: 1.35rem;
        font-weight: 800;
        color: var(--ink);
        margin-bottom: .65rem;
    }

    .chip {
        display: inline-block;
        background: #eff6ff;
        color: #1d4ed8;
        border: 1px solid #bfdbfe;
        border-radius: 999px;
        padding: .18rem .55rem;
        margin: .15rem .2rem .15rem 0;
        font-size: .78rem;
        font-weight: 650;
    }

    .image-placeholder {
        min-height: 210px;
        border-radius: 18px;
        background: linear-gradient(135deg, #e2e8f0, #f8fafc);
        border: 1px dashed #cbd5e1;
        display: flex;
        align-items: center;
        justify-content: center;
        color: var(--muted);
        font-weight: 700;
    }

    div[data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #eef2f7;
        border-radius: 14px;
        padding: .55rem .7rem;
    }

    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid var(--line);
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
        <div class="hero">
          <div class="hero-kicker">Montevideo · Recomendador Inmobil-IA-rio</div>
          <h1>Encuentra una vivienda que calce con tu vida, no solo con tus filtros.</h1>
          <p>
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
        map_col, table_col = st.columns([1.2, 1], gap="large")
        with map_col:
            render_map(response)
        with table_col:
            listings_count = len(response.get("listings_used") or [])
            map_count = len(response.get("map_points") or [])
            response_time = response.get("response_time_sec", "—")
            st.markdown("### Indicadores")
            metric_cols = st.columns(3)
            metric_cols[0].metric("Propiedades", listings_count)
            metric_cols[1].metric("Puntos mapa", map_count)
            metric_cols[2].metric("Tiempo", response_time)
            if response.get("filters_applied"):
                st.markdown("#### Filtros aplicados")
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
