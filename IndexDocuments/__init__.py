import tempfile,os,json,logging,io
import azure.functions as func
from docx import Document
from azure.storage.blob import BlobServiceClient
from langchain.document_loaders import PyPDFLoader
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_text_splitters import CharacterTextSplitter


AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY")
AZURE_STORAGE_CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
AZURE_SEARCH_ENDPOINT = os.environ.get("vector_store_address")
AZURE_SEARCH_KEY = os.environ.get("vector_store_password")
INDEX_NAME = os.environ.get("INDEX_NAME")

def main(req: func.HttpRequest) -> func.HttpResponse:
    try:
        
        if req.method != "POST":
            return func.HttpResponse(
                json.dumps({"status": "FAILED", "error": "Method Not Allowed"}),
                status_code=405,
                mimetype="application/json"
            )

        body = req.get_json()
        doc_link = body.get("doc_link")

        if not doc_link.startswith("https://") or "blob.core.windows.net" not in doc_link:
            raise ValueError("Invalid document link. It must be a Blob Storage URL.")
        
        if not (doc_link.lower().endswith('.pdf') or doc_link.lower().endswith('.docx')):
            return func.HttpResponse(
            json.dumps({
            "status": "FAILED",
            "error": "Invalid or unsupported file link. Only .pdf, .docx, and .doc files are allowed."
        }),
        status_code=400,
        mimetype="application/json"
    )

        file_name = doc_link.split("/")[-1]
        container= doc_link.split("/")[-2]

        blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        blob_client = blob_service_client.get_blob_client(container=container, blob=file_name)
        blob_content = blob_client.download_blob().readall()

        documents = ""
        if doc_link.endswith(".pdf"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                temp_pdf.write(blob_content)
                temp_pdf_path = temp_pdf.name
            
            loader = PyPDFLoader(file_path=temp_pdf_path)
            documents = loader.load()
            
        elif doc_link.endswith(".docx"):
            documents = Document(io.BytesIO(blob_content))
        else:
            raise ValueError("Only PDF and DOCX formats are supported")

        text_splitter = CharacterTextSplitter(chunk_size=1000, separator=" ", chunk_overlap=250)
        docs = text_splitter.split_documents(documents)

        os.remove(temp_pdf_path)

        embeddings = AzureOpenAIEmbeddings(
            azure_deployment="text-embedding-3-small",
            openai_api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY
        )

        vector_store = AzureSearch(
            azure_search_endpoint=AZURE_SEARCH_ENDPOINT,
            azure_search_key=AZURE_SEARCH_KEY,
            index_name=INDEX_NAME,
            embedding_function=embeddings.embed_query
        )

        vector_store.add_documents(docs)

        return func.HttpResponse(
            json.dumps({"status": "COMPLETED", "error": None}),
            status_code=200,
            mimetype="application/json"
        )

    except Exception as e:
        logging.error(f"Error: {e}")
        return func.HttpResponse(
            json.dumps({"status": "FAILED", "error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )
