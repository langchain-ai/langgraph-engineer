import textwrap
from typing import List, Union

from langchain_core.messages import AIMessage, AnyMessage, BaseMessage
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from langgraph.graph import MessageGraph
from langgraph_engineer import ingest

Messages = Union[list[AnyMessage], AnyMessage]


# def add_messages(left: Messages, right: Messages) -> Messages:
#     if not isinstance(left, list):
#         left = [left]
#     if not isinstance(right, list):
#         right = [right]
#     res = left + right
#     return res


# class GraphState(TypedDict):
#     image: Optional[str]
#     messages: Annotated[List[BaseMessage], add_messages]


def wrap_state(state: List[BaseMessage]) -> dict:
    return {"messages": state}


def create_image_interpreter() -> Runnable:

    template = """Here are the full LangGrah docs: \n --- --- --- \n {docs} \n --- --- --- \n 
                You will be shown an image of a graph with nodes as circles and edges \n
                as squares. Each node and edge has a label. Use the provided LangGraph docs to convert \n
                the image into a LangGraph graph. This will have 3 things: (1) create a dummy \n
                state value. (2) Define a dummy function for each each node or edge. (3) finally \n
                create the graph workflow that connects all edges and nodes together. \n
                Structure your answer with a description of the code solution. \n
                Then list the imports. And finally list the functioning code block."""

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert in converting graph visualizations into LangGraph,"
                " a library for building stateful, multi-actor applications with LLMs.\n"
                + textwrap.dedent(template),
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    ).partial(docs=ingest.load_docs())

    # Multi-modal LLM
    model = ChatOpenAI(temperature=0, model="gpt-4-vision-preview", max_tokens="1028")

    return (wrap_state | prompt | model).with_config(run_name="image_to_graph")


# Data model
class code(BaseModel):
    """Code output"""

    module_docstring: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Code block import statements")
    code: str = Field(description="Code block not including import statements")


def format_code(tools: list[code]) -> BaseMessage:
    invoked = tools[0]
    return AIMessage(content=f"\"\"\"\n{invoked.module_docstring}\n\"\"\"\n\n{invoked.imports}\n\n{invoked.code}")


def create_code_formatter() -> Runnable:

    # Structured output prompt
    template = """You are an expert a code formatting, starting with a code solution.

Structure the solution in three parts:
(1) a prefix that defines the problem,
(2) list the imports, and 
(3) list the functioning code block."""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", template),
            MessagesPlaceholder(variable_name="messages"),
            ("system", "Extract the code from the last message and format it."),
        ]
    )

    # LLM
    model = ChatOpenAI(temperature=0, model="gpt-4-0125-preview").bind_tools(
        [code], tool_choice="code"
    )
    # Parser
    parser_tool = PydanticToolsParser(tools=[code])

    def get_last_message(state: List[BaseMessage]) -> dict:
        return {"messages": [state[-1]]}

    return (get_last_message | prompt | model | parser_tool | format_code).with_config(
        run_name="code_formatter"
    )


def create_code_generator() -> Runnable:

    # Structured output prompt
    template = """You are an expert python developer. Develop an application for the user's problem using
LangGraph. Reference the LangGraph docs below for the necessary information. 
<docs>
{docs}
</docs>
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", template),
            MessagesPlaceholder(variable_name="messages"),
        ]
    ).partial(docs=ingest.load_docs())

    # LLM
    model = ChatOpenAI(temperature=0, model="gpt-4-0125-preview").bind_tools(
        [code], tool_choice="code"
    )
    # Parser
    parser_tool = PydanticToolsParser(tools=[code])

    return (format_code | prompt | model | parser_tool | format_code).with_config(
        run_name="code_generator"
    )


def pick_route(state: List[BaseMessage]) -> str:
    message_content = state[-1].content
    if isinstance(message_content, list) and any(
        message["type"] == "image_url" for message in message_content
    ):
        return "understand_image"
    return "generate_code"


def build_graph() -> Runnable:
    builder = MessageGraph()
    builder.add_node(
        "enter",
        lambda _: [],
    )
    builder.add_node("understand_image", create_image_interpreter())
    builder.add_node("format_code", create_code_formatter())
    builder.add_node("generate_code", create_code_generator())

    builder.add_conditional_edges("enter", pick_route)
    builder.add_edge("understand_image", "format_code")
    builder.set_entry_point("enter")
    builder.set_finish_point("generate_code")
    builder.set_finish_point("format_code")
    return builder.compile()
