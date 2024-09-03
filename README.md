# RAG chatbot powered by Langchain and Google Generative AI

## Project Overview <a name="overview"></a>

The aim of this project is to build a RAG chatbot in Langchain powered by [Google Generative AI](https://ai.google.dev/?hl=en)  **APIs**. You can upload documents in txt, pdf, docx or html formats and chat with your data. Relevant documents will be retrieved and sent to the LLM along with your follow-up questions for accurate answers.
Also, developed a user interface using [streamlit](https://streamlit.io/) application to host the chatbot.

## Installation <a name="installation"></a>

This project requires latest Python 3 and the Python libraries which can be found in `requirements.txt`

## Instructions <a name="instructions"></a>

To run the app locally:

1. Clone the github repo: 
2. Migrate to the folder where repo is cloned
3. Create a virtual environment: `conda create -n myenv python=3.12.4`
4. Activate the virtual environment : `conda activate myenv`
4. Install the required dependencies `pip install -r requirements.txt`
5. Start the app: `streamlit run streamlit_app.py`
6. In the sidebar, select the LLM provider (Google Generative AI (default and can be extended to other providers like OpenAI or other free models hosted on Hugging Face)), choose an LLM (Gemini-pro or Gemini-1.5-Flash), adjust its parameters, insert your API key and choose your retriever(Cohere reranker (requires Cohere API key) or Contextual compression or Vectorstore backed retriever).
7. Create a Chroma vectorstore.
8. Chat with your documents: ask questions and get LLM generated answers.

## Deliverables <a name="deliverables"></a>
1. Code Repository: All the relevant python files with relevant comments are added in the repository.
2. Documentation: Information covering system architecture, data ingestion process, rag orchestration and other important aspects can be found at **documentation.pdf**
3. Video Demonstration: Video showcasing the system's capabilities, specifically highlighting the handling of follow-up
questions and quality of question answering. (**video_demonstration.webm**)

