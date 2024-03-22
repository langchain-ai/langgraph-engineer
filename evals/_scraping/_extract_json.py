from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
import json


# class Node(BaseModel):
#     id: str
#     is_start: bool
#     is_end: bool
#     description: str


# class Edge(BaseModel):
#     sources: List[str]
#     targets: List[str]
#     description: str


# class LangGraphSchemaFormat(BaseModel):
#     # graph_state_schema: dict = Field(
#     #     description="The JSON Schema representing the state of the graph (state machine)."
#     # )
#     # nodes: List[Node]
#     edges: List[Edge]


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
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"You are generating JSON for LangGraph representations.\n\n{diagram_format_description}",
        ),
        MessagesPlaceholder(variable_name="convo"),
    ]
)
chain = prompt | ChatOpenAI(model="gpt-4-turbo-preview").with_structured_output(
    None,
    include_raw=True,
    method="json_mode",
)

with open("schemas.jsonl", "r") as f:
    schemas = f.readlines()
    data = [json.loads(schema) for schema in schemas]


def predict_with_retris(schema):
    msgs = [
        (
            "user",
            (
                "Generate the JSON object based on the following text:\n\n<TEXT>\n"
                + schema["predicted"]
                + "\n</TEXT>"
            ),
        )
    ]
    for _ in range(3):
        response = chain.invoke(msgs)
        if response.get("parsing_error") is not None:
            msgs.extend(
                [
                    ("assistant", "... Response omitted ..."),
                    (
                        "user",
                        f'You have an invalid LangGraphSchemaFormat. Error: {response.get("parsing_error")}'
                        "\n\nStrictly adhere to the format described in the prompt.",
                    ),
                ]
            )
        else:
            return response["parsed"]
    raise ValueError("Failed")


results = RunnableLambda(predict_with_retris).batch(data)
dicts = results
# dicts = [result.dict() for result in results]
for raw, schema in zip(data, dicts):
    # schema["state"] = schema.pop("graph_state_schema")
    raw["schema"] = schema
with open("schemas2.json", "w") as f:
    f.write(json.dumps(data, indent=2))
