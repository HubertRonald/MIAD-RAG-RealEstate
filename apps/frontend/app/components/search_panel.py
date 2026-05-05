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
        "title": "Modo 1",
        "subtitle": "Solo filtros estructurados",
        "description": "Ideal cuando el usuario ya conoce zona, presupuesto y tipo de inmueble.",
        "example": "Apartamento en Pocitos, alquiler, 2 dormitorios y presupuesto máximo.",
    },
    {
        "key": "text",
        "title": "Modo 2",
        "subtitle": "Solo texto libre",
        "description": "Ideal para una búsqueda exploratoria escrita en lenguaje natural.",
        "example": "Busco algo tranquilo cerca del mar, con buena luz y terraza.",
    },
    {
        "key": "combined",
        "title": "Modo 3",
        "subtitle": "Texto libre + filtros",
        "description": "Ideal para mezclar intención, contexto familiar y restricciones duras.",
        "example": "Que tenga ascensor y sea moderno, pensando en una familia con niños.",
    },
]

ASK_EXAMPLES = {
    "Precios por zona": "¿Qué diferencia hay entre Pocitos y Punta Carretas en términos de oferta?",
    "Amenities por segmento": "¿Qué amenities son más comunes en apartamentos en alquiler en Carrasco?",
    "Zonas familiares": "¿Qué zonas tienen apartamentos familiares cerca de espacios verdes?",
    "Opciones cerca de la rambla": "¿Qué opciones cerca de la rambla parecen más adecuadas para una familia?",
}


def init_carousel_state() -> None:
    if "recommend_mode_idx" not in st.session_state:
        st.session_state.recommend_mode_idx = 0


def render_mode_carousel() -> int:
    init_carousel_state()
    idx = int(st.session_state.recommend_mode_idx)
    mode = MODE_CONFIG[idx]

    st.markdown(
        f"""
        <div class="mode-carousel">
            <div class="mode-eyebrow">Escenario de búsqueda {idx + 1} de {len(MODE_CONFIG)}</div>
            <h3>{mode['title']} · {mode['subtitle']}</h3>
            <p>{mode['description']}</p>
            <div class="mode-example">{mode['example']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    previous_col, dots_col, next_col = st.columns([1, 2, 1])
    with previous_col:
        if st.button("← Anterior", use_container_width=True, disabled=idx == 0):
            st.session_state.recommend_mode_idx = max(0, idx - 1)
            st.rerun()
    with dots_col:
        dot_cols = st.columns(len(MODE_CONFIG))
        for dot_idx, dot_col in enumerate(dot_cols):
            label = "●" if dot_idx == idx else "○"
            if dot_col.button(label, key=f"mode_dot_{dot_idx}", use_container_width=True):
                st.session_state.recommend_mode_idx = dot_idx
                st.rerun()
    with next_col:
        if st.button("Siguiente →", use_container_width=True, disabled=idx == len(MODE_CONFIG) - 1):
            st.session_state.recommend_mode_idx = min(len(MODE_CONFIG) - 1, idx + 1)
            st.rerun()

    return idx


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
                "Solicitud en lenguaje natural",
                value=MODE_CONFIG[mode_idx]["example"],
                height=110,
                placeholder="Ejemplo: Busco un apartamento luminoso, cerca de la rambla, con terraza y buena conexión.",
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
                help="Si eliges un barrio se envía como string; si eliges varios se envía como arreglo.",
            )
            barrio = _barrio_payload(selected_barrios)

        with st.expander("Configuración avanzada", expanded=False):
            collection = COLLECTION_DEFAULT #st.text_input("collection", value=COLLECTION_DEFAULT)
            max_recommendations = st.slider(
                "Número máximo de recomendaciones",
                min_value=1,
                max_value=5,
                value=DEFAULT_MAX_RECOMMENDATIONS,
            )
            include_map_points = True #st.checkbox("include_map_points", value=True)
            include_explanation = True #st.checkbox("include_explanation", value=True)

        submitted = st.form_submit_button("Mostrarme las mejores propiedades para mí", use_container_width=True)

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


def render_ask_form() -> tuple[bool, dict[str, Any]]:
    with st.form("ask_form", clear_on_submit=False):
        st.markdown("#### Pregunta general sobre el mercado inmobiliario")
        selected_example = st.selectbox("Plantilla rápida", options=list(ASK_EXAMPLES.keys()))
        question = st.text_area(
            "Pregunta",
            value=ASK_EXAMPLES[selected_example],
            height=120,
            placeholder="Ejemplo: ¿Qué zonas tienen apartamentos familiares cerca de espacios verdes?",
        )

        col1, col2 = st.columns(2)
        with col1:
            use_query_rewriting = st.checkbox("Usar query rewriting", value=True)
        with col2:
            use_reranking = st.checkbox("Usar reranking", value=False)

        with st.expander("Configuración avanzada", expanded=False):
            collection = st.text_input("collection", value=COLLECTION_DEFAULT, key="ask_collection")

        submitted = st.form_submit_button("Consultar mercado", use_container_width=True)

    payload = {
        "question": question.strip(),
        "collection": collection or COLLECTION_DEFAULT,
        "use_reranking": use_reranking,
        "use_query_rewriting": use_query_rewriting,
    }
    return submitted, payload
