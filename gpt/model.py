from typing import Optional

from langchain_core.callbacks import BaseCallbackHandler
from streamlit.delta_generator import DeltaGenerator
import streamlit as st

from session.service import save_message_on_session


def create_chat_open_ai(api_key):
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.1,
        streaming=True,
        callbacks=[
            _ChatCallbackHandler(),
        ],
        api_key=api_key
    )


class _ChatCallbackHandler(BaseCallbackHandler):
    message = ""
    message_box: Optional[DeltaGenerator] = None

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message_on_session(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)