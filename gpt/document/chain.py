from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from gpt.document.model import create_chat_open_ai
from gpt.document.prompt import STUFF_PROMPT


def create_chain(retriever, api_key):
    llm = create_chat_open_ai(api_key)
    return {
        'context': retriever | RunnableLambda(_format_docs),
        'question': RunnablePassthrough()
    } | STUFF_PROMPT | llm


def _format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)
