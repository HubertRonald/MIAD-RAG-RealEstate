from __future__ import annotations

from typing import Any

import streamlit as st


def render_debug_panel(
    payload: dict[str, Any] | None,
    response: dict[str, Any] | None,
) -> None:
    if not payload and not response:
        return

    with st.expander("Debug técnico", expanded=False):
        if payload is not None:
            st.markdown("#### Payload enviado")
            st.json(payload)

        if response is not None:
            summary = {
                "collection": response.get("collection"),
                "response_time_sec": response.get("response_time_sec"),
                "filters_applied": response.get("filters_applied"),
                "files_consulted_count": len(response.get("files_consulted") or []),
                "context_docs_count": len(response.get("context_docs") or []),
                "listings_used_count": len(response.get("listings_used") or []),
                "map_points_count": len(response.get("map_points") or []),
                "reranker_used": response.get("reranker_used"),
                "query_rewriting_used": response.get("query_rewriting_used"),
            }

            st.markdown("#### Resumen de respuesta")
            st.json(summary)

            st.markdown("#### Respuesta completa")
            st.json(response)


def render_ask_context(
    response: dict[str, Any],
    *,
    expanded: bool = False,
) -> None:
    final_query = response.get("final_query")
    files_consulted = response.get("files_consulted") or []
    context_docs = response.get("context_docs") or []

    with st.expander("Ver detalle técnico de la respuesta", expanded=expanded):
        col1, col2, col3 = st.columns(3)
        col1.metric("Tiempo", response.get("response_time_sec", "—"))
        col2.metric("Contextos", len(context_docs))
        col3.metric("Archivos", len(files_consulted))

        tab_query, tab_files, tab_contexts = st.tabs(
            ["Query final", "Archivos consultados", "Contextos recuperados"]
        )

        with tab_query:
            if final_query:
                st.code(final_query, language="text")
            else:
                st.caption("No se recibió query final en la respuesta.")

        with tab_files:
            if files_consulted:
                for item in files_consulted:
                    st.write(item)
            else:
                st.caption("No se recibieron archivos/listings consultados.")

        with tab_contexts:
            if context_docs:
                for idx, doc in enumerate(context_docs, start=1):
                    st.markdown(f"#### Contexto {idx}")
                    st.caption(
                        f"file={doc.get('file_name', '—')} · "
                        f"type={doc.get('chunk_type', '—')} · "
                        f"priority={doc.get('priority', '—')} · "
                        f"semantic={doc.get('semantic_score', '—')} · "
                        f"rerank={doc.get('rerank_score', '—')}"
                    )

                    snippet = doc.get("snippet") or doc.get("content") or ""
                    st.write(snippet)

                    if idx < len(context_docs):
                        st.divider()
            else:
                st.caption("No se recibieron contextos recuperados.")
