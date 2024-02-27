# Langgraph-Engineer


A (very alpha) CLI and corresponding notebook for langgraph app generation.

To use, install:

```bash
pip install -U langgraph-engineer
```

You can generate from only a description, or you can pass in a diagram image.

```bash
langgraph-engineer create --description "A RAG app over my local PDF" --diagram "path/to/diagram.png"
```

For example:

```bash
langgraph-engineer create --description "A corrective RAG app" --diagram "CRAG.jpg"
```