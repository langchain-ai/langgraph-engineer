from typing import Iterable
from langchain_core.prompts import ChatPromptTemplate

# from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain.smith import RunEvalConfig
import json
import langsmith


diagram_format_description = """# LangGraph Schema Format

The following describes the intermediate, serialized format of a LangGraph impl

- `state` dict:  JSON schema basically. May include an `annotations` property to provide additional information about the state, with things like freeform descriptions (“append only”) or more strict if we wanted to enable auto code gen w/o an LLM downstream.
- list of `nodes`  - each has an `id` property and optional `is_start` / `is_end` attributes + a `description` of what its purpose and contents are
- list of `edges` - each has a `sources` and  `targets` properties. `sourcse` and `targets` are both lists of strings, indicating the source nodes and target nodes for a given "edge".

If a `condition` property is provided, it contains a description and/or pseudocode outlining the transition logic from source to each of the possible targets.

**Example**:

Below would be a JSON blob representing a simple USER shopping cart processing flow.

```json
{
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
```

Below is an example of a simple multi-agent workflow with a supervisor, coder, and doc writer.

```json
{
  "state": {
    "type": "object",
    "properties": {
      "project": {
        "type": "object",
        "properties": {
          "name": {
            "type": "string",
            "annotations": {
              "description": "Project name",
              "constraints": "Must be a non-empty string"
            }
          },
          "description": {
            "type": "string",
            "annotations": {
              "description": "Project description",
              "constraints": "Must be a non-empty string"
            }
          },
          "requirements": {
            "type": "array",
            "items": {
              "type": "string",
              "annotations": {
                "description": "Project requirement",
                "constraints": "Must be a non-empty string"
              }
            },
            "annotations": {
              "description": "List of project requirements"
            }
          },
          "code": {
            "type": "string",
            "annotations": {
              "description": "Project code",
              "constraints": "Must be valid code"
            }
          },
          "documentation": {
            "type": "string",
            "annotations": {
              "description": "Project documentation",
              "constraints": "Must be a non-empty string"
            }
          }
        },
        "annotations": {
          "description": "Project information"
        }
      }
    },
    "annotations": {
      "description": "Multi-agent workflow state"
    }
  },
  "nodes": [
    {
      "id": "start",
      "is_start": true
    },
    {
      "id": "supervisor_review",
      "description": "Supervisor reviews project requirements"
    },
    {
      "id": "coder_implement",
      "description": "Coder implements the project based on requirements"
    },
    {
      "id": "doc_writer_document",
      "description": "Documentation writer creates project documentation"
    },
    {
      "id": "supervisor_approve",
      "description": "Supervisor approves the project"
    },
    {
      "id": "end",
      "is_end": true
    }
  ],
  "edges": [
    {
      "sources": ["start"],
      "targets": ["supervisor_review"]
    },
    {
      "sources": ["supervisor_review"],
      "targets": ["coder_implement"],
      "condition": "Transition to 'coder_implement' if the supervisor approves the project requirements."
    },
    {
      "sources": ["supervisor_review"],
      "targets": ["end"],
      "condition": "Transition to 'end' if the supervisor rejects the project requirements."
    },
    {
      "sources": ["coder_implement"],
      "targets": ["doc_writer_document"],
    },
    {
      "sources": ["doc_writer_document"],
      "targets": ["supervisor_approve"]
    },
    {
      "sources": ["supervisor_approve"],
      "targets": ["end"],
      "condition": "Transition to 'end' if the supervisor approves the project."
    },
    {
      "sources": ["supervisor_approve"],
      "targets": ["coder_implement"],
      "condition": "Transition back to 'coder_implement' if the supervisor requests changes to the implementation."
    },
    {
      "sources": ["supervisor_approve"],
      "targets": ["doc_writer_document"],
      "condition": "Transition back to 'doc_writer_document' if the supervisor requests changes to the documentation."
    }
  ]
}
```

Note there was no condition for the transition from `coder_implement` to `doc_writer_document` because that transition will ALWAYS occur.
""".replace(
    "{", "{{"
).replace(
    "}", "}}"
)


def parse_json(completion: Iterable[str]):
    state = "outside"
    depth = 0
    start = 0
    i = 0
    result = []
    for chunk in completion:
        for char in chunk:
            result.append(char)
            match state:
                case "outside":
                    if char == "{":
                        state = "inside"
                        depth = 1
                        start = i
                case "inside":
                    if char == "{":
                        depth += 1
                    elif char == "}":
                        depth -= 1
                    if depth == 0:
                        state = "outside"
                        yield json.loads("".join(result[start : i + 1]))
                    elif char == '"':
                        state = "string"
                case "string":
                    if char == "\\":
                        state = "escape"
                    elif char == '"':
                        state = "inside"
                case "escape":
                    state = "string"
            i += 1


llm = ChatOpenAI(model="gpt-4-vision-preview")

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert python engineer tasked with translating software diagrams,"
            " research paper ideas, and other application designs into LangGraph."
            " You will be provided a sketch of the architecture or application and are tasked with generating a JSON blob representing how it could be implemented in LangGraph.\n\n"
            + diagram_format_description,
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

chain = prompt | llm | StrOutputParser() | parse_json


eval_config = RunEvalConfig(evaluators=[])
client = langsmith.Client()


def adapt(inputs: dict):
    images = inputs["images"]
    image_content = [{"type": "image_url", "image_url": image} for image in images]
    return {
        "messages": [
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": "First, think step-by-step about all the optimal state and connectivity "
                        "(which edges will be conditional, which explicit, etc.), then generate the LangGraph representation for this image."
                        " Ensure there are no orphaned states and that every transition is accounted for.",
                    },
                    *image_content,
                ]
            )
        ]
    }


client.run_on_dataset(
    dataset_name="langgraph-diagram2graph",
    llm_or_chain_factory=adapt | chain,
    concurrency_level=1,
)
