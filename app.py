import streamlit as st
import subprocess

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOllama

from uploadFile import UploadFile
from helper.helper import Helper
from prompt import SystemPrompt

def run_google_palm():
    # Configure the API key
    google_api_key = st.secrets["palm_api_key"]

    # Create an instance of the Helper class
    helper = Helper()

    # Check if the API key is set
    if google_api_key == "":
        # Prompt the user to enter the API key
        google_api_key = st.text_input("Google PaLM API Key", type="password")

        # If the user enters an API key, write it to the secrets.toml file.
        if st.button("Set API_KEY") and google_api_key != "":
            helper.set_api_key(google_api_key)
            st.rerun()

    else:
        # If the API key is already set, display a message
        if not st.session_state.api_state_update:
            st.write("API key is set✅")

        # Provide an option to update the API key
        if st.toggle("Update API Key"):
            google_api_key = st.text_input("Enter API key", type="password", key="second")

            # If the user enters a new API key, update it in the secrets.toml file.
            if st.button("Confirm") and google_api_key != "":
                helper.set_api_key(google_api_key)
                st.session_state.api_state_update = True
                st.rerun()

        else:
            st.session_state.api_state_update = False

    # Display a toast message when the API key is updated
    if st.session_state.api_state_update:
        st.toast("API_KEY Updated✅")


def on_embedding_model_change(selected_model):
    if selected_model != st.session_state.embedding_model:
        st.session_state.embedding_model = selected_model
        st.session_state.embedding_model_change_state = True


def select_embedding_model():
    embeddings = ""
    if st.session_state.embedding_model == "HuggingFace Embeddings":
        model_name = "BAAI/bge-base-en-v1.5"
        model_kwargs = {"device": "cpu"}
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name, model_kwargs=model_kwargs
        )
    elif st.session_state.embedding_model == "GooglePalm Embeddings":
        google_api_key = st.secrets["palm_api_key"]
        embeddings = GooglePalmEmbeddings(google_api_key=google_api_key)
    return embeddings


def get_vectorstore(chunks):
    embeddings = select_embedding_model()
    vectorstore = Qdrant.from_documents(
        chunks,
        embeddings,
        location=":memory:",  # Local mode with in-memory storage only
        collection_name="my_documents",
    )
    return vectorstore


def get_conversation(vectorstore):
    if st.session_state.llm_type == "Ollama":
        model = st.session_state.ollama_model
        llm = ChatOllama(
            base_url="http://localhost:11434",
            model=model,
            verbose=True,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
            temperature=0.2,
            num_ctx=512,
            num_thread=16,
            stream=True
        )

    elif st.session_state.llm_type == "Google PaLM":
        google_api_key = st.secrets["palm_api_key"]
        llm = GooglePalm(
            google_api_key=google_api_key, model="chat-bison-001", temperatue=0.2
        )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        return_messages=True,
    )
    prompt = SystemPrompt()
    
    qa = ConversationalRetrievalChain.from_llm(
        llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        memory=memory,
        condense_question_prompt=prompt.questionPrompt(),
    )

    return qa

def process_prompt():
    if prompt := st.chat_input("Ask a question about your documents"):
        st.session_state.chat_dialog_history.append({"role": "user", "content": prompt})
    # Display the prior chat messages
    for message in st.session_state.chat_dialog_history:
        with st.chat_message(name=message["role"]):
            st.markdown(message["content"])
    if st.session_state.chat_dialog_history[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.conversation({"question": prompt})
                st.markdown(response["answer"])
                st.session_state.chat_history = response["chat_history"]
                message = {"role": "assistant", "content": response["answer"]}
                st.session_state.chat_dialog_history.append(message)

def load_models():
    LLM_TYPES = ["Google PaLM", "Ollama"]
    # OLLAMA_MODELS = ["Mistral 7B", "EverythingLM 13B", "Orca-Mini 3B"]
    EMBEDDING_MODELS = ["HuggingFace Embeddings", "GooglePalm Embeddings"]

    # Checking the available ollama models
    try:
        ollama_list_output = subprocess.check_output(["ollama", "list"]).decode().split("\n")
    except Exception:
        try:
            ollama_list_output = subprocess.check_output(["docker", "exec", "-it", "ollama", "ollama", "list"]).decode().split("\n")
        except Exception:
            ollama_list_output = []

    OLLAMA_MODELS = [line.split()[0] for line in ollama_list_output if ":" in line and "ollama:" not in line]
    
    model_type = st.selectbox("Select LLM ⬇️", LLM_TYPES)
    if model_type == "Google PaLM":
        run_google_palm()
    elif model_type == "Ollama":
        if not OLLAMA_MODELS:
            st.error("Ollama is not configured properly, Make sure:\n\n"
                    "1. You have installed Ollama.\n"
                    "2. Ollama is running.\n"
                    "3. You have downloaded an Ollama model like Mistral 7B.")
            st.session_state.error=True
        else:
            st.session_state.ollama_model = st.selectbox("Ollama Model", OLLAMA_MODELS)
    st.session_state.llm_type = model_type
    # handling the embedding models
    embedding_model = st.radio("Embedding Model ⬇️", EMBEDDING_MODELS)
    on_embedding_model_change(embedding_model)


def load_ui():
    st.set_page_config(
        page_title="Fileo AI :books:",
        page_icon=":books:",
        layout="wide",
        initial_sidebar_state="auto",
        menu_items=None,
    )

    st.title("Fileo AI :books:")
    # checking the session state for the conversation
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_dialog_history" not in st.session_state.keys():
        st.session_state.chat_dialog_history = [
            {"role": "assistant", "content": "Hello!, Ask me your queries 🤔"}
        ]
    # using for the check of uploaded documents
    if "disabled" not in st.session_state:
        st.session_state.disabled = True
    # using for api state check
    if "api_state_update" not in st.session_state:
        st.session_state.api_state_update = False
    if "ollama_model" not in st.session_state:
        st.session_state.ollama_model = ""
    if "embedding_model" not in st.session_state:
        st.session_state.embedding_model = " "
    if "llm_type" not in st.session_state:
        st.session_state.llm_type = "Google PaLM"
    if "llm" not in st.session_state:
        st.session_state.llm = dict()
    if "embedding_model_change_state" not in st.session_state:
        st.session_state.embedding_model_change_state = False
    if "error" not in st.session_state:
        st.session_state.error = False
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = []
    if "messages" not in st.session_state:
        st.session_state.messages = []

def process_document():
    with st.sidebar:
        load_models()
        st.info("⚠️ Ensure Docs are processed again on change of Model configurations")
        # for the documents
        text_chunks = []
        if pdf_docs := st.file_uploader(
            "Upload the PDFs here:", accept_multiple_files=True,
            type=["png","jpg","jpeg","xlsx","xls","csv","pptx","docx","pdf","txt"]
        ):
            if st.button("Process", type="primary", use_container_width=True, disabled=st.session_state.error):
                with st.spinner("Processing..."):
                    if (
                        st.session_state.disabled
                        or st.session_state.embedding_model_change_state
                    ):
                        for pdf in pdf_docs:
                            upload = UploadFile(pdf)
                            splits = upload.get_document_splits()
                            text_chunks.extend(splits)
                        st.session_state.vectorstore = get_vectorstore(text_chunks)
                        st.session_state.embedding_model_change_state = False
                    # create conversation chain
                    # we need to make the conversation chain persistent as streamlit reruns
                    # the code when a button is pressed
                    st.session_state.conversation = get_conversation(
                        st.session_state.vectorstore
                    )
                    # Update the session state to enable the text input
                    st.toast("The processing was successful! Ask away!", icon="✅")
                    st.session_state.disabled = False
    # st.write(text_chunks)
    if st.session_state.disabled:
        st.write("🔒 Please upload and process your PDFs to unlock the question field.")

    else:
        process_prompt()


def main():
    load_ui()
    process_document()


if __name__ == "__main__":
    main()
