from langchain.agents import create_structured_chat_agent, AgentExecutor
from langchain import hub
from langchain_core.language_models import BaseLanguageModel

from agents.langchains.tools import LANGCHAIN_SEARCH_TOOLS

_PROMPT = hub.pull("hwchase17/structured-chat-agent")


class LangChainChatAgentExecutor:
    def __init__(self, llm_model: BaseLanguageModel):
        self._llm_model = llm_model
        self._agent = create_structured_chat_agent(
            llm=llm_model,
            tools=LANGCHAIN_SEARCH_TOOLS,
            prompt=_PROMPT
        )
        self._executor = AgentExecutor(
            agent=self._agent,
            tools=LANGCHAIN_SEARCH_TOOLS,
        )

    def search(self, query: str) -> str:
        invoked_result = self._executor.invoke({
            "input": query
        })
        return invoked_result['output']
