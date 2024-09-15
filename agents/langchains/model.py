from langchain_openai import ChatOpenAI


def create_langchain_agent_openai_model(api_key: str) -> ChatOpenAI:
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        api_key=api_key
    )
