import os
import tempfile
import requests
from langchain_community.document_loaders import NotebookLoader
from langchain.load.dump import dumps
from tqdm.auto import tqdm
import base64
import mimetypes
import re

# Required by the notebook loader. Fail early
import pandas as pd  # type: ignore

repo_url = "https://api.github.com/repos/langchain-ai/langgraph/contents"
access_token = os.environ["GITHUB_ACCESS_TOKEN"]

# Set the headers for authentication (if required)
headers = {"Authorization": f"token {access_token}"}

all_docs = []


def load_image_base64(url: str, headers: dict) -> str:
    # Load a URL as a base64 image
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        # Convert the image to base64
        mime_type = mimetypes.guess_type(url)[0]
        base64_image = base64.b64encode(response.content).decode("utf-8")
        return f"data:{mime_type};base64,{base64_image}"
    else:
        print(f"Failed to load image from {url}. Status code: {response.status_code}")
        return ""


def get_notebook_files(url):
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        contents = response.json()
        for item in tqdm(contents):
            if item["type"] == "dir":
                # Recursively search subdirectories
                get_notebook_files(item["url"])
            elif item["name"].endswith(".ipynb"):
                # Download the notebook file to a temporary file
                notebook_url = item["download_url"]
                notebook_name = item["name"]
                print(f"Downloading notebook: {notebook_name}")

                notebook_content = requests.get(notebook_url).content
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".ipynb"
                ) as temp_file:
                    temp_file.write(notebook_content)
                    temporary_file = temp_file.name

                # Load the notebook using NotebookLoader
                docs = NotebookLoader(temporary_file, remove_newline=True).load()
                doc = docs[0]
                all_docs.append(doc)
                doc.metadata["url"] = item["url"]
                # Add a regex to extract the path from markdown image renders like ![foobazz.](path)
                regex = r"!\[.*\]\((.*)\)"
                matches = re.findall(regex, doc.page_content)
                if len(matches) > 0:
                    resolved_matches = [
                        # Merge url with the relative path if applicable, normalizing the path (removing any "./", etc.)
                        os.path.normpath(
                            os.path.join(os.path.dirname(item["url"]), match)
                        )
                        .replace("https:/api", "https://api")
                        .replace("api.github.com/repos", "raw.githubusercontent.com")
                        .replace("/contents", "/main", 1)
                        for match in matches
                        if not match.startswith("http")
                        and not match.startswith("https")
                        and not match.startswith("attachment:")
                    ]
                    doc.metadata["image_paths"] = resolved_matches
                    imgs = list(
                        filter(
                            bool,
                            [
                                load_image_base64(match, headers)
                                for match in resolved_matches
                            ],
                        )
                    )
                    doc.metadata["images"] = imgs

                # Clean up the temporary file
                os.unlink(temporary_file)
    else:
        print(
            f"Failed to retrieve contents from {url}. Status code: {response.status_code}"
        )
    return all_docs


# Start the recursive search from the repository root
docs = get_notebook_files(repo_url)
with open("scraped_docs.json", "w") as f:
    f.write(dumps(docs))

print(f"Total loaded documents: {len(all_docs)}")
