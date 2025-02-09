import os,json,logging
import azure.functions as func
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential

def initialize_openai_client():
    
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    
    if not endpoint or not api_key or not api_version:
        raise ValueError("One or more environment variables are missing or invalid.")
    
    embedding_client = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)
    return embedding_client

def initialize_search_client(index_name):
    
    endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    api_key = os.getenv("AZURE_SEARCH_KEY")
    
    if not endpoint or not api_key:
        raise ValueError("One or more environment variables for Azure Search are missing or invalid.")
    
    search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=AzureKeyCredential(api_key))
    return search_client

def generate_embedding(embedding_client, text):
    emb = embedding_client.embeddings.create(model="text-embedding-3-small", input=text)
    res = json.loads(emb.model_dump_json())
    return res["data"][0]["embedding"]  

def perform_vector_search(search_client, query_vector):
    
    search_parameters = {
        
        "vector": {
            "value": query_vector,
            "fields": "content_vector",
            "k": 10  
        },
    }

    search_results = search_client.search(search_parameters)
    return search_results

def create_system_message(content_chunks):
    system_message = os.getenv("SYSTEM_MESSAGE_TEMPLATE")
    content_message = "\n".join(content_chunks)
    return f"{system_message}\n\nContent:\n{content_message}"

def main(req: func.HttpRequest) -> func.HttpResponse:
    try:
        if req.method != "GET":
            return func.HttpResponse(
                json.dumps({"response": None, "error": "Method Not Allowed"}),
                status_code=405,
                mimetype="application/json"
            )
        body=req.get_body
        query=[query]
        query = req.params.get("query")
        index_name = req.params.get("index_name")

        if not query or not index_name:
            return func.HttpResponse(
                json.dumps({"response": None, "error": "Missing required query parameters: 'query' and 'index_name'"}),
                status_code=400,
                mimetype="application/json"
            )

        embedding_client = initialize_openai_client()
        search_client = initialize_search_client(index_name)

        query_vector = generate_embedding(embedding_client, query)

        search_results = perform_vector_search(search_client, query_vector)

        content_chunks = [result['content'] for result in search_results]
        if content_chunks:
            system_message = create_system_message(content_chunks)
        else:
            system_message = os.getenv("SYSTEM_MESSAGE_TEMPLATE")
            
        message_text=[{"role":"system","content":system_message},{"role":"user","content":query}]

        response = embedding_client.chat.completions.create(
            model="gpt-35-turbo-16k",
            messages=message_text,temperature=0.25
        )

        gpt_response = response.choices[0].message.content.replace('\n','')

        if gpt_response:
            return func.HttpResponse(
                json.dumps({"response": gpt_response, "error": None}),
                status_code=200,
                mimetype="application/json"
            )
        else:
            return func.HttpResponse(
                json.dumps({"response": None, "error": None}),
                status_code=200,
                mimetype="application/json"
            )

    except ValueError as ve:
        logging.error(f"Configuration error: {str(ve)}")
        return func.HttpResponse(
            json.dumps({"response": None, "error": str(ve)}),
            status_code=500,
            mimetype="application/json"
        )
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return func.HttpResponse(
            json.dumps({"response": None, "error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )
