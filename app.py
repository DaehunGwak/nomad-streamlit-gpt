import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

from gpt.document.chain import create_chain
from session.service import save_message_on_session, init_message_on_session
from view.cache import get_cached_retriever_after_embedding
from view.message import paint_message, paint_message_history

if "messages" not in st.session_state:
    init_message_on_session()

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)
st.title("DocumentGPT")
st.markdown("""
---
### Welcome!
- Use this chatbot to ask questions  to an AI about your files
- Upload your file on the sidebar.
---
""")

ctx = get_script_run_ctx()

with st.sidebar:
    st.subheader("Inputs")
    input_api_key = st.text_input("Input your api key")
    file = st.file_uploader("Upload your .txt file", type=['txt'])

    st.subheader("Features")
    if st.button("Clear your chat histories"):
        init_message_on_session()

    st.markdown("---")
    st.subheader("Debugs")
    st.write(f"session uid: {ctx.session_id}")
    st.write(f"api key: {input_api_key}")


if file and input_api_key:
    retriever = get_cached_retriever_after_embedding(file, ctx.session_id, input_api_key)

    if len(st.session_state["messages"]) <= 0:
        save_message_on_session("i'm ready! Ask away!", "ai")
    paint_message_history()

    input_message = st.chat_input("Ask anything about your file")
    if input_message:
        save_message_on_session(input_message, "human")
        paint_message(input_message, "human")

        chain = create_chain(retriever, input_api_key)
        with st.chat_message("ai"):
            response = chain.invoke(input_message)

else:
    init_message_on_session()
