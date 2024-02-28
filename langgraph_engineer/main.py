import logging
from pathlib import Path
from typing import List, Optional

import typer
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.utils import image as image_utils
from langgraph.graph import END
from langgraph_engineer import ingest, system
from typing_extensions import Annotated


logging.basicConfig(level=logging.INFO)

app = typer.Typer(no_args_is_help=True, add_completion=True)


@app.command(name="create")
def create(
    description: str = typer.Argument(
        ..., help="Description of the application to be created."
    ),
    diagram: Annotated[
        Optional[Path],
        typer.Option(
            help="Path to the image file to be used as the base for the graph"
        ),
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option(
            help="Path to the file where the graph should be saved. Default is stdout.",
        ),
    ] = None,
):
    """
    Create a graph from an image file.
    """
    graph_ = system.build_graph()
    if diagram:
        image = image_utils.image_to_data_url(str(diagram))
        content = [{"type": "image_url", "image_url": image}]
    content.append({"type": "text", "text": description})
    last_chunk = None
    for chunk in graph_.stream(HumanMessage(content=content)):
        typer.echo(f"Running step {next(iter(chunk))}...")
        last_chunk = chunk
    code_content = ""
    if last_chunk:
        messages: List[BaseMessage] = last_chunk[END]
        code_content = messages[-1].content
    if output:
        with output.open("w") as f:
            f.write(code_content)
    else:
        typer.echo(code_content)


@app.command(name="ingest")
def ingest_docs(
    dry_run: bool = typer.Option(
        False, help="Print the ingested documents instead of writing them to file."
    )
):
    """
    Ingest a file into the graph.
    """
    ingest.ingest(dry_run=dry_run)


if __name__ == "__main__":
    app()
