import json
from langchain.load import loads
import langsmith

with open("schemas2.json", "r") as f:
    extracted = loads(f.read())

mapped = {e["url"]: e for e in extracted}


with open("scraped_docs.json", "r") as f:
    scraped = loads(f.read())

omapped = {doc.metadata["url"]: doc.metadata.get("images") for doc in scraped}

# Now create schemas 3 as the mapped list but add the images in
schemas3 = []
for k, v in mapped.items():
    v["images"] = omapped[k]
    schemas3.append(v)


# Write out
with open("schemas3.json", "w") as f:
    f.write(json.dumps(schemas3))

client = langsmith.Client()

dataset_name = "langgraph-diagram2graph"
if client.has_dataset(dataset_name=dataset_name):
    client.delete_dataset(dataset_name=dataset_name)
ds = client.create_dataset(dataset_name=dataset_name)
client.create_examples(
    inputs=[{"images": e["images"]} for e in schemas3],
    outputs=[e["schema"] for e in schemas3],
    dataset_id=ds.id,
)