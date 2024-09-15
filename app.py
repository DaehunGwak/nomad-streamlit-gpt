import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

from agents.langchains.executor import LangChainChatAgentExecutor
from agents.langchains.model import create_langchain_agent_openai_model
from session.service import save_message_on_session, init_message_on_session
from view.message import paint_message, paint_message_history

if "messages" not in st.session_state:
    init_message_on_session()

st.set_page_config(
    page_title="OpenAI Assistants - Search : Ordi",
    page_icon="🔍",
)
st.title("🔍 OpenAI Assistants")
st.info("""
### 👋 Welcome! 
- Input your OpenAI key on the sidebar.
- Use this agents to search with DuckDuckGo & Wikipedia
""")

ctx = get_script_run_ctx()

with st.sidebar:
    st.subheader("Inputs")
    input_api_key = st.text_input("Input your api key")

    st.subheader("Features")
    if st.button("Clear your chat histories"):
        init_message_on_session()

    st.markdown("---")
    st.subheader("Debugs")
    st.write(f"session uid: {ctx.session_id}")
    st.write(f"api key: {input_api_key}")


if input_api_key:
    model = create_langchain_agent_openai_model(input_api_key)
    executor = LangChainChatAgentExecutor(llm_model=model)

    if len(st.session_state["messages"]) <= 0:
        save_message_on_session("i'm ready! Ask away!", "ai")
    paint_message_history()

    input_message = st.chat_input("Ask something to search on DuckDuckGo or Wikipedia")
    if input_message:
        save_message_on_session(input_message, "human")
        paint_message(input_message, "human")

        output = executor.search(query=input_message)
        save_message_on_session(output, "ai")
        paint_message(output, "ai")

else:
    init_message_on_session()
