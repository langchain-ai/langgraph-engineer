import json
from langchain.load import loads
from langchain_core.prompts import ChatPromptTemplate
from tqdm import tqdm

# from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.prompts import MessagesPlaceholder

filename = "scraped_docs.json"
with open(filename, "r") as f:
    docs = loads(f.read())

with_images = [doc for doc in docs if doc.metadata.get("images")]


diagram_format_description = """# LangGraph Schema Format

The following describes the intermediate, serialized format of a LangGraph impl

- `state` dict:  JSON schema basically. May include an `annotations` property to provide additional information about the state, with things like freeform descriptions (“append only”) or more strict if we wanted to enable auto code gen w/o an LLM downstream.
- list of `nodes`  - each has an `id` property and optional `is_start` / `is_end` attributes + a `description` of what its purpose and contents are
- list of `edges` - each has a `sources` and  `targets` properties. `sourcse` and `targets` are both lists of strings, indicating the source nodes and target nodes for a given "edge".
If a `condition` property is provided, it contains a description and/or pseudocode outlining the transition logic from source to each of the possible targets.

**Example**:

```json

  "state": {
    "type": "object",
    "properties": {
      "user": {
        "type": "object",
        "properties": {
          "name": {
            "type": "string",
            "annotations": {
              "description": "User's full name",
              "constraints": "Must contain at least 2 words"
            }
          },
          "email": {
            "type": "string",
            "format": "email",
            "annotations": {
              "description": "User's email address",
              "constraints": "Must be a valid email format"
            }
          }
        },
        "annotations": {
          "description": "User information"
        }
      },
      "items": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "id": {
              "type": "string",
              "annotations": {
                "description": "Unique item identifier",
                "constraints": "Must be a UUID"
              }
            },
            "quantity": {
              "type": "integer",
              "annotations": {
                "description": "Item quantity",
                "constraints": "Must be a positive integer"
              }
            }
          },
          "annotations": {
            "description": "List of items"
          }
        }
      }
    },
    "annotations": {
      "description": "User's shopping cart"
    }
  },
  "nodes": [
    {
      "id": "start",
      "is_start": true
    },
    {
      "id": "process"
    },
    {
      "id": "end",
      "is_end": true
    }
  ],
  "edges": [
    {
      "sources": ["start"],
      "targets": ["process"]
    },
    {
      "sources": ["process"],
      "targets": ["end"],
      "condition": "Transition to 'end' after processing the user's shopping cart."
    }
  ]
}
```""".replace(
    "{", "{{"
).replace(
    "}", "}}"
)

system_prompt = f"""You are tasked with generating JSON blobs representing LangGraph implementations based on\
the example notebooks and diagrams.

<LangGraph Description>
LangGraph is a library for building stateful, multi-actor applications with LLMs, built on top of (and intended to be used with) LangChain. It extends the LangChain Expression Language with the ability to coordinate multiple chains (or actors) across multiple steps of computation in a cyclic manner. It is inspired by Pregel and Apache Beam. The current interface exposed is one inspired by NetworkX.

The main use is for adding cycles to your LLM application. Crucially, LangGraph is NOT optimized for only DAG workflows. If you want to build a DAG, you should just use LangChain Expression Language.

Cycles are important for agent-like behaviors, where you call an LLM in a loop, asking it what action to take next.
{diagram_format_description}
</LangGraph Description>"""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "{instructions}"),
        MessagesPlaceholder(variable_name="messages"),
    ]
).partial(instructions=system_prompt)
llm = ChatOpenAI(model="gpt-4-vision-preview")


def format_doc(doc: Document) -> list:
    # Return list of the form:
    content = [
        {
            "type": "text",
            "text": "Generate the JSON intermediate representation"
            f" based on the following notebook:\n\n<notebook>\n{doc.page_content}\n\n</notebook>\n",
        },
    ]
    for image in doc.metadata.get("images", []):
        content.append(
            {
                "type": "image_url",
                "image_url": image,
            }
        )
    content.append(
        {
            "type": "text",
            "text": "Generate valid LangGraph Schema JSON to represent the application.",
        }
    )
    return {"messages": [("user", content)]}


chain = format_doc | prompt | llm | StrOutputParser()

with open("schemas.jsonl", "w") as f:
    for i, schema in tqdm(chain.batch_as_completed(with_images)):
        url = with_images[i].metadata["url"]
        f.write(json.dumps({"url": url, "predicted": schema, "i": i}) + "\n")
