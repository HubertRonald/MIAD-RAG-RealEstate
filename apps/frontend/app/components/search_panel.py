from __future__ import annotations

from typing import Any

import streamlit as st

from utils.formatting import normalize_barrio

COLLECTION_DEFAULT = "realstate_mvd"
DEFAULT_MAX_RECOMMENDATIONS = 3

BARRIOS = [
    "Pocitos",
    "Punta Carretas",
    "Carrasco",
    "Buceo",
    "Malvín",
    "Parque Rodó",
    "Cordón",
    "Centro",
    "Ciudad Vieja",
    "Aguada",
    "Tres Cruces",
    "La Blanqueada",
    "Punta Gorda",
]

MODE_CONFIG = [
    {
        "key": "filters",
        "title": "Solo filtros",
        "subtitle": "Elige zona, precio y características desde opciones predefinidas.",
        "description": "Ideal si ya tienes claro el barrio, presupuesto y tipo de inmueble.",
        "example": "Pocitos · 2 dorm. · alquiler",
        "icon": "🔎",
        "chips": ["Pocitos", "2 dorm.", "Alquiler"],
        "recommended": False,
    },
    {
        "key": "text",
        "title": "Solo texto",
        "subtitle": "Describe lo que buscas con tus propias palabras.",
        "description": "Útil para búsquedas exploratorias, como si le contaras a un asesor.",
        "example": "Busco algo tranquilo cerca del mar, con buena luz y terraza.",
        "icon": "📝",
        "chips": ["Cerca del mar", "Terraza", "Buena luz"],
        "recommended": False,
    },
    {
        "key": "combined",
        "title": "Texto y filtros",
        "subtitle": "Combina una descripción libre con filtros para afinar resultados.",
        "description": "Recomendado si quieres explicar tu necesidad y además limitar precio, zona o dormitorios.",
        "example": "Quiero algo luminoso con balcón.",
        "icon": "✨",
        "chips": ["hasta USD 120k", "2 dorm.", "Balcón"],
        "recommended": True,
    },
]

ASK_SUGGESTIONS = [
    {
        "label": "Precios por zona",
        "question": "¿Qué diferencia hay entre Pocitos y Punta Carretas en términos de oferta?",
    },
    {
        "label": "Amenities por segmento",
        "question": "¿Qué amenities son más comunes en apartamentos en alquiler en Carrasco?",
    },
    {
        "label": "Zonas familiares",
        "question": "¿Qué zonas tienen apartamentos familiares cerca de espacios verdes?",
    },
    {
        "label": "Cerca de la rambla",
        "question": "¿Qué opciones cerca de la rambla parecen más adecuadas para una familia?",
    },
    {
        "label": "Alquiler vs. compra",
        "question": "¿Qué diferencias hay entre apartamentos en alquiler y venta por zona?",
    },
    {
        "label": "Pregunta libre",
        "question": "",
    },
]


def init_mode_state() -> None:
    if "recommend_mode_idx" not in st.session_state:
        st.session_state.recommend_mode_idx = 0


def render_mode_selector_cards() -> int:
    init_mode_state()

    current_idx = int(st.session_state.recommend_mode_idx)

    st.markdown("### ¿Cómo prefieres buscar tu próxima propiedad?")

    cols = st.columns(3, gap="medium")

    for idx, mode in enumerate(MODE_CONFIG):
        selected = idx == current_idx

        with cols[idx]:
            with st.container(border=True):
                if selected:
                    st.markdown(
                        "<div class='selected-card-accent'></div>",
                        unsafe_allow_html=True,
                    )

                top_col1, top_col2 = st.columns([1, 2])

                with top_col1:
                    st.markdown(
                        f"<div class='mode-icon-box'>{mode['icon']}</div>",
                        unsafe_allow_html=True,
                    )

                with top_col2:
                    if mode.get("recommended"):
                        st.markdown(
                            "<span class='recommended-badge'>Recomendado</span>",
                            unsafe_allow_html=True,
                        )

                st.markdown(f"#### {mode['title']}")
                st.caption(mode["subtitle"])

                st.markdown(
                    f"""
                    <div class="mode-example-box">
                        “{mode["example"]}”
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                chips_html = "".join(
                    f"<span class='mode-chip'>{chip}</span>"
                    for chip in mode["chips"]
                )
                st.markdown(chips_html, unsafe_allow_html=True)

                button_label = "Seleccionado" if selected else "Elegir"

                if st.button(
                    button_label,
                    key=f"mode_card_select_{idx}",
                    use_container_width=True,
                    type="primary" if selected else "secondary",
                ):
                    st.session_state.recommend_mode_idx = idx
                    st.rerun()

    return current_idx


def render_mode_carousel() -> int:
    """Alias temporal por compatibilidad. El carrusel se reemplazó por tarjetas."""
    return render_mode_selector_cards()


def _barrio_payload(selected: list[str]) -> str | list[str] | None:
    barrios = [normalize_barrio(b) for b in selected if b]

    if not barrios:
        return None

    if len(barrios) == 1:
        return barrios[0]

    return barrios


def _add_if_present(payload: dict[str, Any], key: str, value: Any) -> None:
    if value in (None, "", [], {}):
        return
    payload[key] = value


def render_recommend_form(mode_idx: int) -> tuple[bool, dict[str, Any]]:
    mode_key = MODE_CONFIG[mode_idx]["key"]
    show_text = mode_key in {"text", "combined"}
    show_filters = mode_key in {"filters", "combined"}

    with st.form(key=f"recommend_form_{mode_key}", clear_on_submit=False):
        st.markdown("#### Cuéntanos qué propiedad estás buscando")

        question = ""

        if show_text:
            question = st.text_area(
                "Describe qué estás buscando",
                value=MODE_CONFIG[mode_idx]["example"],
                height=110,
                placeholder=(
                    "Ejemplo: busco un apartamento luminoso, cerca de la rambla, "
                    "con terraza y buena conexión."
                ),
            )

        operation_type = None
        property_type = None
        barrio = None
        max_price = None
        min_bedrooms = None

        if show_filters:
            col1, col2 = st.columns(2)

            with col1:
                operation_type = st.selectbox(
                    "Tipo de operación",
                    options=["alquiler", "venta"],
                    index=0,
                )

                max_price = st.number_input(
                    "Presupuesto máximo",
                    min_value=0,
                    value=0,
                    step=10000,
                    help="Déjalo en 0 para no enviar este filtro.",
                )

            with col2:
                property_type = st.selectbox(
                    "Tipo de inmueble",
                    options=["apartamentos", "casas"],
                    index=0,
                )

                min_bedrooms = st.number_input(
                    "Dormitorios mínimos",
                    min_value=0,
                    value=0,
                    step=1,
                    help="Déjalo en 0 para no enviar este filtro.",
                )

            selected_barrios = st.multiselect(
                "Barrios de interés",
                options=BARRIOS,
                default=["Pocitos"],
                help="Puedes seleccionar uno o varios barrios.",
            )
            barrio = _barrio_payload(selected_barrios)

        with st.expander("Configuración avanzada", expanded=False):
            collection = COLLECTION_DEFAULT

            max_recommendations = st.slider(
                "Número máximo de recomendaciones",
                min_value=1,
                max_value=5,
                value=DEFAULT_MAX_RECOMMENDATIONS,
            )

            include_map_points = True
            include_explanation = True

        submitted = st.form_submit_button(
            "Ver recomendaciones",
            use_container_width=True,
            type="primary",
        )

    payload: dict[str, Any] = {
        "collection": collection or COLLECTION_DEFAULT,
        "max_recommendations": max_recommendations,
        "include_map_points": include_map_points,
        "include_explanation": include_explanation,
    }

    if show_text:
        _add_if_present(payload, "question", question.strip())

    if show_filters:
        _add_if_present(payload, "operation_type", operation_type)
        _add_if_present(payload, "property_type", property_type)
        _add_if_present(payload, "barrio", barrio)

        if max_price and max_price > 0:
            payload["max_price"] = int(max_price)

        if min_bedrooms and min_bedrooms > 0:
            payload["min_bedrooms"] = int(min_bedrooms)

    return submitted, payload


def _init_ask_state() -> None:
    if "ask_question_input" not in st.session_state:
        st.session_state["ask_question_input"] = ASK_SUGGESTIONS[0]["question"]


def render_ask_form() -> tuple[bool, dict[str, Any]]:
    _init_ask_state()

    st.markdown("#### Consulta sobre el mercado inmobiliario de Montevideo")
    st.caption("Elige un tema o escribe tu propia pregunta.")

    chip_cols = st.columns(3, gap="small")
    for idx, suggestion in enumerate(ASK_SUGGESTIONS):
        with chip_cols[idx % 3]:
            if st.button(
                f"• {suggestion['label']}",
                key=f"ask_suggestion_{idx}",
                use_container_width=True,
            ):
                st.session_state["ask_question_input"] = suggestion["question"]
                st.rerun()

    with st.form("ask_form", clear_on_submit=False):
        question = st.text_area(
            "Pregunta",
            key="ask_question_input",
            height=95,
            placeholder="Escribe tu pregunta sobre el mercado...",
            label_visibility="collapsed",
        )

        st.caption(
            "El asistente responde preguntas sobre propiedades del mercado montevideano. "
            "No consulta precios en tiempo real ni datos externos."
        )

        submitted = st.form_submit_button(
            "Consultar",
            use_container_width=True,
            type="primary",
        )

    payload = {
        "question": (question or "").strip(),
        "collection": COLLECTION_DEFAULT,
        "use_reranking": False,
        "use_query_rewriting": False,
    }

    return submitted, payload