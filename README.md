# Azure Function App using Vector Search

This project implements two Azure Function APIs:

1. **IndexDocuments**: Accepts a document link (PDF or DOCX) and indexes it into Azure AI search.
2. **QueryKnowledgeBase**: Queries the indexed documents and retrieves relevant information using OpenAI's GPT model.

## Features

### **IndexDocuments**:
- Reads documents from Azure Blob Storage.
- Splits document content into chunks.
- Generates embeddings using OpenAI's `text-embedding-3-small` model.
- Indexes chunks into Azure AI search.
- Handles errors and returns a status message.

### **QueryKnowledgeBase**:
- Takes a user query and converts it into a vector embedding.
- Searches through the indexed content using vector search.
- Uses GPT to return relevant information or null if no relevant information is found.

## Technologies Used
- **Azure Functions**
- **Azure AI Search**
- **Azure OpenAI**
- **Python**
- **Postman**

## Environment Variables
Ensure the following environment variables are set:
- `AZURE_OPENAI_ENDPOINT`: Azure OpenAI API endpoint.
- `AZURE_OPENAI_API_KEY`: Azure OpenAI API key.
- `AZURE_OPENAI_API_VERSION`: OpenAI API version.
- `AZURE_SEARCH_ENDPOINT`: Azure Search API endpoint.
- `AZURE_SEARCH_KEY`: Azure Search API key.
- `SYSTEM_MESSAGE_TEMPLATE`: The system message template for GPT responses.

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/ShashankGowni/Azure-Function-App-using-Vector-search
    ```

2. Navigate to the project folder:
    ```bash
    cd Azure-Function-App-using-Vector-search
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Set the necessary environment variables:
    - For local development, create a `.env` file or set them in the terminal.

## Running Locally
To run the function app locally, use the Azure Functions Core Tools:
```bash
func start
```

## Output
### **API Usage**

1. **IndexDocuments API**
   - **Request Type**: POST  
   - **Endpoint**: `http://localhost:<port>/api/IndexDocuments`
   - **Request Body**:
     ```json
     {
       "doc_link": "https://yourblobstorageurl.com/yourfile.pdf"
     }
     ```
   - **Success Response**:
     ```json
     {
       "status": "COMPLETED",
       "error": null
     }
     ```
   - **Failure Response**:
     ```json
     {
       "status": "FAILED",
       "error": "Document format is not supported."
     }
     ```

2. **QueryKnowledgeBase API**
   - **Request Type**: GET  
   - **Endpoint**: `http://localhost:<port>/api/QueryKnowledgeBase?query=What%20is%20Azure&index_name=your_index_name`
   - **Success Response**:
     ```json
     {
       "response": "Azure is a cloud computing service from Microsoft.",
       "error": null
     }
     ```
   - **Failure Response**:
     ```json
     {
       "response": null,
       "error": null
     }
     ```

## **Output Screenshot (References)**
![API Output](https://github.com/user-attachments/assets/7d25b9e0-a9d6-4a2e-971f-ba7b93730e68)
![API Output](https://github.com/user-attachments/assets/7d25b9e0-a9d6-4a2e-971f-ba7b93730e68)
