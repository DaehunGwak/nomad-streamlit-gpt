from datetime import timedelta

import streamlit as st

from gpt.document.retriever import get_retriever_after_embedding


@st.cache_resource(show_spinner="Embedding file...", ttl=timedelta(hours=1))
def get_cached_retriever_after_embedding(input_file, session_id, api_key):
    return get_retriever_after_embedding(input_file, session_id, api_key)
