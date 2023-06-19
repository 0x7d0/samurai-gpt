from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
import os
import glob
from typing import List
import requests
from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from constants import CHROMA_SETTINGS
from pydantic import BaseModel

# ... (Remaining imports and class definitions, as well as the 'load_single_document', 'load_documents', 'LOADER_MAPPING', and 'download_and_save' functions)

class Query(BaseModel):
    text: str

app = FastAPI()

load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
llm = None

class MyElmLoader(UnstructuredEmailLoader):
    """Wrapper to fallback to text/plain when default does not work"""

    # ... (Remaining class definition)

# ... (Remaining class definitions and functions)

@app.get('/ingest')
async def ingest_data():
    # ... (Same function body as Flask app)

@app.post('/get_answer')
async def get_answer(query: Query):
    # ... (Same function body as Flask app, but replace 'request.json' with 'query.text')

@app.post('/upload_doc')
async def upload_doc(document: UploadFile = File(...)):
    # ... (Same function body as Flask app, but replace 'request.files' with 'document')

@app.get('/download_model')
async def download_and_save():
    # ... (Same function body as Flask app)

def load_model():
    filename = 'ggml-gpt4all-j-v1.3-groovy.bin'  # Specify the name for the downloaded file
    models_folder = 'models'  # Specify the name of the folder inside the Flask app root
    file_path = f'{models_folder}/{filename}'
    if os.path.exists(file_path):
        global llm
        callbacks = [StreamingStdOutCallbackHandler()]
        llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', callbacks=callbacks, verbose=False)

if __name__ == "__main__":
  load_model()
  print("LLM0", llm)
  import uvicorn
  uvicorn.run(app, host="0.0.0.
