import streamlit as st
import os
from data_ingestion import delete_temp_files, langchain_document_loader, split_documents_to_chunks, TMP_DIR
from vector_store import select_embeddings_model, create_vectorstore, create_retriever
from rag_system import create_ConversationalRetrievalChain

# configs

list_LLM_providers = [
    "**Google Generative AI**",
]

dict_welcome_message = {
    "English": "How can I assist you today?",
}

list_retriever_types = [
    "Cohere reranker",
    "Contextual compression",
    "Vectorstore backed retriever",
]

LOCAL_VECTOR_STORE_DIR = "./vector_stores"
if not os.path.exists(TMP_DIR):
    os.makedirs(TMP_DIR)
if not os.path.exists(LOCAL_VECTOR_STORE_DIR):
    os.makedirs(LOCAL_VECTOR_STORE_DIR)

st.set_page_config(page_title="Chat With Your Data")
def reset_session():
    st.session_state.clear()
    st.rerun()

st.title("🤖 RAG chatbot")

if st.button("Reset Session"):
    reset_session()
# st.title("🤖 RAG chatbot")

# API keys
st.session_state.google_api_key = ""
st.session_state.cohere_api_key = ""
st.session_state.hf_api_key = ""
st.session_state.llm_api_key = ""
    
def expander_model_parameters(
    llm_provider="Google",
    text_input_API_key="Google API Key - [Get an API key](https://makersuite.google.com/app/apikey)",
    list_models=["gemini-pro"],captions = ["Rate limit: 60 requests per minute."]
):
    """Add a text_input (for API key) and a streamlit expander containing models and parameters."""
    st.session_state.LLM_provider = llm_provider

    if llm_provider == "Google":
        st.session_state.google_api_key = st.text_input(
            text_input_API_key,
            type="password",
            placeholder="insert your API key",
        )
        st.session_state.llm_api_key = st.session_state.google_api_key
        st.session_state.hf_api_key = ""


    with st.expander("**Models and parameters**"):
        # st.session_state.selected_model = st.selectbox(
        #     f"Choose {llm_provider} model", list_models)
        st.session_state.selected_model = st.radio(
            f"Choose {llm_provider} model", list_models,captions=captions)
        # model parameters
        st.session_state.temperature = st.slider("temperature",min_value=0.0,max_value=1.0,value=0.5,step=0.1)
        st.session_state.top_p = st.slider("top_p",min_value=0.0,max_value=1.0,value=0.95,step=0.05)


def sidebar():
    """Create the sidebar and the a tabbed pane: the first tab contains a document chooser (create a new vectorstore);
    the second contains a vectorstore chooser (open an old vectorstore)."""

    with st.sidebar:
        st.caption(
            "🚀 A retrieval augmented generation chatbot powered by 🔗 Langchain, Cohere, Google Generative AI"
        )
        st.write("")

        llm_chooser = st.radio("Select provider",list_LLM_providers,
            captions=["Rate limit: 15 requests per minute."],)

        st.divider()
        if llm_chooser == list_LLM_providers[0]:
            expander_model_parameters(
                llm_provider="Google",
                text_input_API_key="Google API Key - [Get an API key](https://makersuite.google.com/app/apikey)",
                list_models=["gemini-pro", "gemini-1.5-flash"],
                captions=["Rate limit: 60 requests per minute.", "Rate limit: 15 requests per minute."],
            )
        # Assistant language
        st.write("")
        st.session_state.assistant_language = st.selectbox(
            f"Assistant language", list(dict_welcome_message.keys())
        )

        st.divider()
        st.subheader("Retrievers")
        retrievers = list_retriever_types
        st.session_state.retriever_type = st.selectbox(
            f"Select retriever type", retrievers
        )
        st.write("")
        if st.session_state.retriever_type == list_retriever_types[0]:  # Cohere
            st.session_state.cohere_api_key = st.text_input(
                "Coher API Key - [Get an API key](https://dashboard.cohere.com/api-keys)",
                type="password",
                placeholder="insert your API key",
            )

        st.write("\n\n")
        st.write(
            f"ℹ _Your {st.session_state.LLM_provider} API key, '{st.session_state.selected_model}' parameters, \
            and {st.session_state.retriever_type} are only considered when loading or creating a vectorstore._"
        )

def documentChooser():
    # 1. Select documents
    st.session_state.uploaded_file_list = st.file_uploader(
        label="**Select documents**",
        accept_multiple_files=True,
        type=(["pdf", "txt", "docx", "html"]),
    )
    # 2. Process documents
    st.session_state.vector_store_name = st.text_input(
        label="**Documents will be loaded, embedded and ingested into a vectorstore (Chroma dB). Please provide a valid dB name.**",
        placeholder="Vectorstore name",
    )
    st.session_state.vector_store_name = str(st.session_state.vector_store_name)
    # 3. Add a button to process documnets and create a Chroma vectorstore
    col1, col2 = st.columns([4, 2])
    with col1:
        if st.button("Create Vectorstore"):
            if "vector_store" in st.session_state:
                st.warning(f"Vectorstore {st.session_state.vector_store_name} is already created for the session. If you want to create a new vectorstore, reload the session.")
            else:
                chain_RAGBlocks()
    with col2:
        if "is_vectorstore" not in st.session_state:
            st.session_state.is_vectorstore = True
        if st.button("Update Vectorstore", help="Press the button after you create a vectorstore", disabled=st.session_state.is_vectorstore):
            error_messages = create_error_messages()
            if "vector_store" not in st.session_state:
                error_messages.append("Vectorstore not created")
            if len(error_messages) >= 1:
                st.session_state.error_message = "Please " + ", ".join(error_messages)
            else: # 2. Upload selected documents to temp directory
                st.session_state.error_message = ""
                delete_temp_files(TMP_DIR)  # 1. Delete old temp files
                file_names = []
                for uploaded_file in st.session_state.uploaded_file_list:
                    error_message = ""
                    file_names.append(uploaded_file.name)
                    try:
                        temp_file_path = os.path.join(TMP_DIR, uploaded_file.name)
                        with open(temp_file_path, "wb") as temp_file:
                            temp_file.write(uploaded_file.read())
                    except Exception as e:
                        error_message += e
                if error_message != "":
                    st.warning(f"Errors: {error_message}")

                # 3. Load documents with Langchain loaders
                documents = langchain_document_loader()
                st.info(f"Documents " + ",".join(file_names) + " uploaded successfully")
                # 4. Split documents to chunks
                chunks = split_documents_to_chunks(documents)
                st.session_state.vector_store.add_documents(chunks)
                st.success(f"Vectorstore **{st.session_state.vector_store_name}** updated succussfully.")

    try:
        if st.session_state.error_message != "":
            st.warning(st.session_state.error_message)
    except:
        pass


def create_error_messages():
    error_messages = []
    if (not st.session_state.google_api_key and not st.session_state.hf_api_key):
        error_messages.append(f"insert your {st.session_state.LLM_provider} API key")

    if (st.session_state.retriever_type == list_retriever_types[0] and not st.session_state.cohere_api_key):
        error_messages.append(f"insert your Cohere API key")

    if not st.session_state.uploaded_file_list:
        error_messages.append("select documents to upload")

    if st.session_state.vector_store_name == "":
        error_messages.append("provide a Vectorstore name")
    return error_messages

def chain_RAGBlocks():
    """The RAG system is composed of:
    - 1. Retrieval: includes document loaders, text splitter, vectorstore and retriever.
    - 2. Memory.
    - 3. Converstaional Retreival chain.
    """
    with st.spinner("Creating vectorstore..."):
        # Check inputs
        error_messages = create_error_messages()
        if len(error_messages) == 1:
            st.session_state.error_message = "Please " + error_messages[0] + "."
        elif len(error_messages) > 1:
            st.session_state.error_message = ("Please "+ ", ".join(error_messages[:-1])+ ", and "+ error_messages[-1] +".")
        else:
            st.session_state.error_message = ""
            try:
                delete_temp_files(TMP_DIR)  # 1. Delete old temp files
                file_names=[]
                if st.session_state.uploaded_file_list is not None: # 2. Upload selected documents to temp directory
                    for uploaded_file in st.session_state.uploaded_file_list:
                        error_message = ""
                        file_names.append(uploaded_file.name)
                        try:
                            temp_file_path = os.path.join(TMP_DIR, uploaded_file.name)
                            with open(temp_file_path, "wb") as temp_file:
                                temp_file.write(uploaded_file.read())
                        except Exception as e:
                            error_message += e
                    if error_message != "":
                        st.warning(f"Errors: {error_message}")

                    # 3. Load documents with Langchain loaders
                    documents = langchain_document_loader()
                    st.info(f"Documents " + ",".join(file_names) + " uploaded successfully")
                    # 4. Split documents to chunks
                    chunks = split_documents_to_chunks(documents)
                    # 5. Embeddings
                    embeddings = select_embeddings_model(llm_api_key=st.session_state.llm_api_key,
                                                        llm_provider= st.session_state.LLM_provider)
                    st.session_state.embeddings = embeddings

                    # 6. Create a vectorstore
                    try:
                        st.session_state.vector_store, new_vectorstore_name = create_vectorstore(
                            embeddings=embeddings,
                            documents = chunks,
                            vectorstore_name=st.session_state.vector_store_name)
                        st.info(f"Vectorstore **{st.session_state.vector_store_name}** is created succussfully.")
                       
                         # 7. Create retriever
                        st.session_state.retriever = create_retriever(
                            vector_store=st.session_state.vector_store,
                            embeddings=st.session_state.embeddings,
                            retriever_type=st.session_state.retriever_type,
                            base_retriever_search_type="similarity",
                            base_retriever_k=16,
                            compression_retriever_k=20,
                            cohere_api_key=st.session_state.cohere_api_key,
                            cohere_model="rerank-multilingual-v2.0",
                            cohere_top_n=10,
                        )

                        # 8. Create memory and ConversationalRetrievalChain
                        st.session_state.chain, st.session_state.memory= create_ConversationalRetrievalChain(
                            retriever=st.session_state.retriever,
                            chain_type="stuff",
                            language=st.session_state.assistant_language,
                            llm_provider = st.session_state.LLM_provider,
                            llm_api_key=st.session_state.llm_api_key,
                            llm_selected_model=st.session_state.selected_model,
                            temperature=st.session_state.temperature,
                            top_p=st.session_state.top_p)

                        # 9. Clear chat_history
                        # clear_chat_history()
                        st.success("Chatbot Tab is ready to use.")
                        st.session_state.is_vectorstore = False
                    except Exception as e:
                        st.error(f"Error:{e}")

            except Exception as error:
                st.error(f"An error occurred: {error}")

def clear_chat_history():
    """clear chat history and memory."""
    # 1. re-initialize messages
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": dict_welcome_message[st.session_state.assistant_language],
        }
    ]
    # 2. Clear memory (history)
    try:
        st.session_state.memory.clear()
    except:
        pass

def get_response_from_LLM(prompt):
    """invoke the LLM, get response, and display results (answer and source documents)."""
    try:
        # 1. Invoke LLM
        response = st.session_state.chain.invoke({"question": prompt})
        answer = response["answer"]

        if st.session_state.LLM_provider == "HuggingFace":
            answer = answer[answer.find("\nAnswer: ") + len("\nAnswer: ") :]

        # 2. Display results
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            # 2.1. Display anwser:
            st.markdown(answer)

    except Exception as e:
        st.warning(e)

def chatbot():
    sidebar()
    # Tabbed Pane: Create a new Vectorstore | Open a saved Vectorstore

    document_chooser, chat_bot = st.tabs(
        ["Create a new Vectorstore", "Chatbot"]
    )
    with document_chooser:
    # 1. Select documents
        documentChooser()
        st.write("\n\n")
        st.write("**Note:**" + " If you want to change model or model parameters after creating the vectorstore either click on " + "*Reset Session*" + 
                 " or hard reload the webpage")
        # st.divider()
    with chat_bot:
        if "is_vectorstore" in st.session_state and not st.session_state.is_vectorstore:
            col1, col2 = st.columns([7, 3])
            with col1:
                st.subheader("Chat with your data")
            with col2:
                st.button("Clear Chat History", on_click=clear_chat_history)

            if "messages" not in st.session_state:
                st.session_state["messages"] = [
                    {
                        "role": "assistant",
                        "content": dict_welcome_message[st.session_state.assistant_language],
                    }
                ]
            for msg in st.session_state.messages:
                st.chat_message(msg["role"]).write(msg["content"])

            if prompt := st.chat_input("Ask a question about yourself"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.chat_message("user").write(prompt)
                with st.spinner("Running..."):
                    get_response_from_LLM(prompt=prompt)
        else:
            st.subheader("Chat with your data")
            st.chat_message("assistant").write("Apologies, the chatbot is unavailable as the vector store hasn’t been created yet.")
                
chatbot()

