from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import Tool

_WIKIPEDIA_TOOL = Tool(
    name="Wikipedia",
    func=WikipediaAPIWrapper().run,
    description="Use this to search for information on Wikipedia."
)
_DUCKDUCKGO_TOOL = Tool(
    name="duckduckgo search",
    func=DuckDuckGoSearchRun().run,
    description="Use this to search the web using DuckDuckGo."
)

LANGCHAIN_SEARCH_TOOLS = [_WIKIPEDIA_TOOL, _DUCKDUCKGO_TOOL]

