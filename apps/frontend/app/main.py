from __future__ import annotations

import os
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from components.debug_panel import render_ask_context, render_debug_panel
from components.map_view import render_map
from components.property_cards import (
    render_answer_block,
    render_listings_table,
    render_property_cards,
)
from components.search_panel import (
    render_ask_form,
    render_mode_selector_cards,
    render_recommend_form,
)
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

.hero-card {
  background:
    linear-gradient(135deg, rgba(219, 234, 254, 0.9), rgba(255, 255, 255, 0.96)),
    linear-gradient(135deg, rgba(249, 115, 22, 0.08), transparent);
  border: 1px solid var(--border);
  border-radius: 24px;
  padding: 2rem;
  box-shadow: 0 18px 45px rgba(15, 23, 42, 0.08);
  margin-bottom: 1.5rem;
}

.hero-eyebrow {
  color: var(--orange);
  font-weight: 800;
  letter-spacing: .08em;
  text-transform: uppercase;
  font-size: .78rem;
}

.mode-card {
  min-height: 280px;
  background: #ffffff;
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 1rem;
  box-shadow: 0 12px 30px rgba(15, 23, 42, 0.05);
  margin-bottom: .55rem;
}

.mode-card-selected {
  border: 2px solid var(--orange);
  box-shadow: 0 16px 38px rgba(249, 115, 22, 0.15);
}

.mode-card-top {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.mode-icon {
  width: 38px;
  height: 38px;
  border-radius: 12px;
  background: var(--blue-soft);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.15rem;
}

.mode-title {
  margin-top: .9rem;
  font-size: 1.05rem;
  font-weight: 800;
  color: var(--navy);
}

.mode-subtitle {
  margin-top: .45rem;
  color: var(--muted);
  font-size: .92rem;
  line-height: 1.45;
}

.mode-example {
  margin-top: .85rem;
  background: #f8fafc;
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: .75rem;
  font-size: .86rem;
  color: #334155;
}

.mode-chip-row {
  margin-top: .8rem;
}

.mode-chip, .soft-chip {
  display: inline-block;
  background: var(--orange-soft);
  color: #9a3412;
  border: 1px solid #fed7aa;
  border-radius: 999px;
  padding: .2rem .55rem;
  margin: .15rem;
  font-size: .75rem;
  font-weight: 700;
}

.recommended-badge {
  background: var(--orange);
  color: #ffffff;
  border-radius: 999px;
  padding: .22rem .55rem;
  font-size: .72rem;
  font-weight: 800;
}

.answer-card {
  background: #ffffff;
  border: 1px solid var(--border);
  border-left: 5px solid var(--orange);
  border-radius: 18px;
  padding: 1.25rem;
  box-shadow: 0 12px 35px rgba(15, 23, 42, 0.06);
}

.rank-pill {
  display: inline-block;
  background: #172033;
  color: #ffffff;
  border-radius: 999px;
  padding: .25rem .6rem;
  font-weight: 800;
  font-size: .8rem;
  margin-bottom: .5rem;
}

.image-placeholder {
  min-height: 220px;
  background: #e7e5dc;
  color: #64748b;
  border-radius: 14px;
  display: flex;
  flex-direction: column;
  gap: .4rem;
  align-items: center;
  justify-content: center;
  font-weight: 700;
}

.image-placeholder-icon {
  font-size: 2rem;
  opacity: .65;
}

.property-meta {
  color: #64748b;
  font-size: .8rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: .04em;
}

.property-title {
  color: var(--navy);
  font-size: 1.08rem;
  font-weight: 800;
  margin-top: .25rem;
}

.property-price {
  color: #0f172a;
  font-size: 1.35rem;
  font-weight: 900;
  margin-top: .25rem;
}

.search-chip-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: .75rem;
  flex-wrap: wrap;
  margin: 1rem 0 .7rem 0;
}

.search-chip-group {
  display: flex;
  gap: .35rem;
  flex-wrap: wrap;
}

.search-count {
  font-size: .92rem;
  font-weight: 800;
  color: var(--navy);
}

.stButton > button {
  border-radius: 12px;
}

div[data-testid="stMetric"] {
  background: #ffffff;
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: .7rem;
}

.loading-clean-slate {
  min-height: 880px;
  width: 100%;
  background:
    linear-gradient(180deg, rgba(248, 251, 255, 0.98), rgba(248, 251, 255, 1));
  border-top: 1px solid #dbe4f0;
  margin-top: 0.75rem;
  padding-top: 1rem;
}

.loading-message {
  display: flex;
  align-items: center;
  gap: 0.65rem;
  color: #172033;
  font-size: 0.98rem;
  font-weight: 600;
}

.loading-spinner {
  width: 20px;
  height: 20px;
  border: 3px solid #dbeafe;
  border-top: 3px solid #f97316;
  border-radius: 50%;
  animation: appSpin 0.8s linear infinite;
}

@keyframes appSpin {
  to {
    transform: rotate(360deg);
  }
}

.property-header-row {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 1rem;
  width: 100%;
  margin-bottom: 0.6rem;
}

.property-header-main {
  flex: 1;
  min-width: 0;
}

.score-badge {
  display: inline-block;
  background: #ecfccb;
  color: #3f6212;
  border-radius: 999px;
  padding: 0.28rem 0.65rem;
  font-size: 0.78rem;
  font-weight: 800;
  white-space: nowrap;
  flex-shrink: 0;
  align-self: flex-start;
}

.stat-card {
  background: #f7f5ef;
  border: 1px solid #ece8dd;
  border-radius: 10px;
  padding: 0.7rem 0.55rem;
  min-height: 72px;
  text-align: center;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}

.stat-label {
  color: #172033;
  font-size: 0.78rem;
  font-weight: 600;
  line-height: 1.1;
  margin-bottom: 0.28rem;
  white-space: nowrap;
  text-align: center;
}

.stat-value {
  color: #172033;
  font-size: 1.15rem;
  font-weight: 800;
  line-height: 1.1;
  white-space: nowrap;
  overflow: visible;
  text-overflow: clip;
  text-align: center;
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
    st.sidebar.text_input(
        "BACKEND_URL",
        value=backend_url or "No configurado",
        disabled=True,
    )
    st.sidebar.caption(f"Auth mode: `{os.getenv('BACKEND_AUTH_MODE', 'auto')}`")

    if st.sidebar.button(
        "Probar /health",
        use_container_width=True,
        disabled=client is None,
    ):
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
    st.sidebar.caption(
        "El frontend no replica lógica analítica: solo arma payloads, "
        "llama al backend y renderiza la respuesta."
    )

    return page


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero-card">
          <p class="hero-eyebrow">
            Montevideo · Recomendador Inmobil-IA-rio
          </p>
          <h1>Su Casa Ya: Te ayudamos a encontrar la vivienda de tus sueños ¡dinos qué estás buscando!</h1>
          <p style="color:#475569; font-size:1rem; max-width:850px;">
            Prototipo para probar recomendaciones para compra o alquiler de
            casas o apartamentos en Montevideo, Uruguay, con destino a vivienda propia.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _clear_recommendation_results() -> None:
    st.session_state.pop("last_recommend_payload", None)
    st.session_state.pop("last_recommend_response", None)
    st.session_state.pop("show_all_listings", None)


def _clear_ask_results() -> None:
    st.session_state.pop("last_ask_payload", None)
    st.session_state.pop("last_ask_response", None)


def _format_chip_value(value: Any) -> list[str]:
    if value in (None, "", [], {}):
        return []

    if isinstance(value, list):
        return [str(v).replace("_", " ").title() for v in value if v]

    return [str(value).replace("_", " ").title()]


def render_search_summary_chips(
    payload: dict[str, Any] | None,
    listings_count: int,
) -> None:
    if not payload:
        return

    chips: list[str] = []

    for barrio in _format_chip_value(payload.get("barrio")):
        chips.append(barrio)

    if payload.get("min_bedrooms"):
        chips.append(f"{payload['min_bedrooms']} dorm.")

    if payload.get("operation_type"):
        chips.append(str(payload["operation_type"]).title())

    if payload.get("max_price"):
        chips.append(f"hasta {payload['max_price']:,}".replace(",", "."))

    if payload.get("property_type"):
        chips.append(str(payload["property_type"]).title())

    chips_html = "".join(
        f"<span class='mode-chip'>{chip}</span>"
        for chip in chips
    )

    st.markdown(
        f"""
        <div class="search-chip-bar">
          <div class="search-chip-group">{chips_html}</div>
          <div class="search-count">{listings_count} propiedades encontradas</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_availability_notice(
    response: dict[str, Any],
    payload: dict[str, Any] | None,
) -> None:
    listings = response.get("listings_used") or []
    found = len(listings)

    requested = 0
    if payload:
        try:
            requested = int(payload.get("max_recommendations") or 0)
        except (TypeError, ValueError):
            requested = 0

    if found == 0:
        st.warning(
            "Ups. Lamentablemente no encontramos propiedades para ti. "
            "Por favor, danos otra oportunidad probando con una nueva búsqueda."
        )
        return

    if requested and found < requested:
        st.info(
            f"Ups. Sólo encontramos {found} "
            f"{'propiedad' if found == 1 else 'propiedades'} para ti. "
            "Puedes probar una nueva búsqueda ampliando zona, presupuesto o características."
        )


def render_recommendation_results(
    response: dict[str, Any],
    last_payload: dict[str, Any] | None,
) -> None:
    listings = response.get("listings_used") or []
    listings_count = len(listings)

    # Caso sin resultados:
    # No mostrar mapa, chips, cards, tabla ni resumen del asistente.
    # Solo mostrar el mensaje amigable de búsqueda sin resultados.
    if listings_count == 0:
        render_result_availability_notice(response, last_payload)
        return

    # Caso con resultados:
    # Mostrar chips, mapa, aviso si llegaron menos de los solicitados,
    # cards y resumen opcional.
    render_search_summary_chips(last_payload, listings_count)

    render_map(
        response,
        show_title=False,
        height=335,
        show_points_debug=False,
    )

    render_result_availability_notice(response, last_payload)

    render_property_cards(listings)
    
    # Debug técnico desactivado para usuario final.
    # render_listings_table(listings)

    # Mantener oculto por defecto, solo cuando sí hay resultados.
    with st.expander("Presiona aquí para ver el resumen del asistente", expanded=False):
        render_answer_block(response, title=None)

    # Debug técnico desactivado para usuario final.
    # render_debug_panel(last_payload, response)


def render_loading_clean_slate(message: str) -> None:
    st.markdown(
        f"""
        <div class="loading-clean-slate">
          <div class="loading-message">
            <div class="loading-spinner"></div>
            <div>{message}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def run_recommendation_page(client: BackendClient | None) -> None:
    st.markdown("## Prueba el nuevo recomendador")

    mode_idx = render_mode_selector_cards()
    submitted, payload = render_recommend_form(mode_idx)

    recommendation_results_slot = st.empty()

    if submitted:
        if client is None:
            st.error("Configura BACKEND_URL antes de consultar.")
            return

        _clear_recommendation_results()

        try:
            with recommendation_results_slot.container():
                render_loading_clean_slate(
                    "Estamos buscando las mejores opciones para vos."
                )

            response = client.recommend(payload)

            st.session_state["last_recommend_payload"] = payload
            st.session_state["last_recommend_response"] = response

            recommendation_results_slot.empty()

            with recommendation_results_slot.container():
                render_recommendation_results(response, payload)

        except Exception as exc:  # noqa: BLE001
            recommendation_results_slot.empty()
            with recommendation_results_slot.container():
                st.error(str(exc))

        return

    response = st.session_state.get("last_recommend_response")
    last_payload = st.session_state.get("last_recommend_payload")

    if response:
        with recommendation_results_slot.container():
            render_recommendation_results(response, last_payload)


def render_ask_results(
    response: dict[str, Any],
    last_payload: dict[str, Any] | None,
) -> None:
    st.markdown("### Respuesta")
    st.markdown(
        f"""
        <div class="answer-card">
            {response.get("answer", "Sin respuesta")}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # render_ask_context(response, expanded=False)
    # render_debug_panel(last_payload, response)


def run_ask_page(client: BackendClient | None) -> None:
    submitted, payload = render_ask_form()

    ask_results_slot = st.empty()

    if submitted:
        if client is None:
            st.error("Configura BACKEND_URL antes de consultar.")
            return

        if not payload.get("question"):
            st.warning("Escribe una pregunta o selecciona una sugerencia.")
            return

        _clear_ask_results()

        try:
            with ask_results_slot.container():
                render_loading_clean_slate(
                    "Consultando el mercado inmobiliario..."
                )

            response = client.ask(payload)

            st.session_state["last_ask_payload"] = payload
            st.session_state["last_ask_response"] = response

            ask_results_slot.empty()

            with ask_results_slot.container():
                render_ask_results(response, payload)

        except Exception as exc:  # noqa: BLE001
            ask_results_slot.empty()
            with ask_results_slot.container():
                st.error(str(exc))

        return

    response = st.session_state.get("last_ask_response")
    last_payload = st.session_state.get("last_ask_payload")

    if response:
        with ask_results_slot.container():
            render_ask_results(response, last_payload)


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
