import streamlit as st


def paint_message(message, role):
    with st.chat_message(role):
        st.markdown(message)


def paint_message_history():
    for message in st.session_state["messages"]:
        paint_message(message["message"], message["role"])
