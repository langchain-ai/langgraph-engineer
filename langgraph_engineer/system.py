import functools
import textwrap
from typing import List, Union

from langchain_core.messages import AIMessage, AnyMessage, BaseMessage
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from langgraph.graph import END, MessageGraph
from langgraph_engineer import code_utils, ingest

Messages = Union[list[AnyMessage], AnyMessage]


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


def format_code(tools: list[code], name: str = "Junior Developer") -> BaseMessage:
    invoked = tools[0]
    return AIMessage(
        content=f'"""\n{invoked.module_docstring}\n"""\n\n{invoked.imports}\n\n{invoked.code}',
        name=name,
    )


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


def lint_code(state: List[BaseMessage]) -> List[BaseMessage]:
    synthetic_code = state[-1].content
    res = code_utils.run_ruff(synthetic_code)
    if res["error"]:
        result = [
            AIMessage(
                content=f"{res['error']}\n\nOutput:\n{res['out']}", name="Code Reviewer"
            )
        ]
    else:
        result = []
    return result


def should_regenerate(state: List[BaseMessage], max_tries: int = 3) -> str:
    num_code_reviewer_messages = sum(
        1 for message in state if message.name == "Code Reviewer"
    )
    if (
        num_code_reviewer_messages == num_code_reviewer_messages
        or num_code_reviewer_messages >= max_tries
    ):
        # Either no errors or too many attempts
        return END

    return "fix_code"


def create_code_fixer() -> Runnable:
    template = """You are an expert python developer, knowledgeable in LangGraph.
Fix the junior developer's draft code to ensure it is free of errors.
Consult the following docs for the necessary information.
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

    llm = ChatOpenAI(temperature=0, model="gpt-4-0125-preview").bind_tools(
        [code], tool_choice="code"
    )
    parser = PydanticToolsParser(tools=[code])

    def format_messages(state: List[BaseMessage]):
        # Remove any images here
        messages = []
        for message in state:
            if isinstance(message.content, str):
                messages.append(message)
                continue
            if any(message["type"] == "image_url" for message in message.content):
                new_content = [
                    content
                    for content in message.content
                    if content["type"] != "image_url"
                ]
                messages.append(
                    message.__class__(
                        **message.dict(exclude={"content"}), content=new_content
                    )
                )
                continue
            messages.append(message)
        return {"messages": messages}

    return (
        format_messages
        | prompt
        | llm
        | parser
        | functools.partial(format_code, name="Senior Developer")
    ).with_config(run_name="code_fixer")


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
    builder.add_node("lint_code", lint_code)
    builder.add_node("fix_code", create_code_fixer())

    builder.add_conditional_edges("enter", pick_route)
    builder.add_edge("understand_image", "format_code")
    builder.add_edge("generate_code", "lint_code")
    builder.add_edge("format_code", "lint_code")
    builder.add_edge("fix_code", "lint_code")
    builder.set_entry_point("enter")
    builder.add_conditional_edges("lint_code", should_regenerate)
    return builder.compile()
