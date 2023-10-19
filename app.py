import streamlit as st
import toml

from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS, Qdrant
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from langchain.memory import ConversationBufferMemory

import google.generativeai as palm
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm


def set_api_key(google_api_key):
    with open(".streamlit/secrets.toml", "r+") as f:
        secrets = toml.load(f)
        secrets["palm_api_key"] = google_api_key
        f.seek(0)
        toml.dump(secrets, f)
        f.truncate()


def run_google_palm():
    google_api_key = st.secrets["palm_api_key"]
    palm.configure(api_key=google_api_key)
    # If the API key is not set in the secrets.toml file, prompt the user to enter it.
    if google_api_key == "":
        google_api_key = st.text_input("Google PaLM API Key", type="password")
        # If the user enters an API key, write it to the secrets.toml file.
        if st.button("Set API_KEY"):
            # google_api_key = st.text_input("Google PaLM API Key", type='password',key='first')
            if google_api_key != "":
                set_api_key(google_api_key)
                st.session_state.api_state_set = True
                st.rerun()
    else:
        if st.session_state.api_state_set:
            st.write("API key is set✅")
            st.session_state.api_state_set = False

        if st.toggle("Update API_KEY"):
            google_api_key = st.text_input(
                "Enter API key", type="password", key="second"
            )
            if st.button("Update API_KEY"):
                # google_api_key = st.text_input("Google PaLM API Key", type='password',key='first')
                if google_api_key != "":
                    set_api_key(google_api_key)
                    st.session_state.api_state_update = True
                    st.rerun()
        else:
            st.session_state.api_state_update = False
    if st.session_state.api_state_update:
        st.toast("API_KEY Updated✅")


def on_embedding_model_change(selected_model):
    if selected_model != st.session_state.embedding_model:
        st.session_state.embedding_model = selected_model
        st.session_state.embedding_model_change_state = True


def load_models():
    LLM_TYPES = ["Google PaLM", "Ollama"]
    OLLAMA_MODELS = ["Mistral 7B", "EverythingLM 13B", "Orca-Mini 3B"]
    EMBEDDING_MODELS = ["HuggingFace Embeddings", "GooglePalm Embeddings"]

    model_type = st.selectbox("Select LLM ⬇️", LLM_TYPES)
    if model_type == "Google PaLM":
        run_google_palm()
    elif model_type == "Ollama":
        st.session_state.ollama_model = st.radio("Ollama Model", OLLAMA_MODELS)
    st.session_state.llm_type = model_type
    # handling the embedding models
    embedding_model = st.radio("Embedding Model ⬇️", EMBEDDING_MODELS)
    on_embedding_model_change(embedding_model)


def load_ui():
    st.set_page_config(
        page_title="ChatPDF",
        page_icon=":books:",
        layout="wide",
        initial_sidebar_state="auto",
        menu_items=None,
    )

    st.title("ChatPDF :books:")
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
    if "api_state_set" not in st.session_state:
        st.session_state.api_state_set = False
    if "ollama_model" not in st.session_state:
        st.session_state.ollama_model = "Mistral 7B"
    if "embedding_model" not in st.session_state:
        st.session_state.embedding_model = " "
    if "llm_type" not in st.session_state:
        st.session_state.llm_type = "Google PaLM"
    if "llm" not in st.session_state:
        st.session_state.llm = dict()
    if "embedding_model_change_state" not in st.session_state:
        st.session_state.embedding_model_change_state = False


def get_text_chunks(pdf_docs):
    chunks = list()
    for pdf in pdf_docs:
        text = ""
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
        text = text.encode("ascii", "ignore").decode("ascii")

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1500,
            chunk_overlap=200,
            length_function=len,
        )
        docs = text_splitter.create_documents([text])
        for i, doc in enumerate(docs):
            doc.metadata = {"source": f"source_{i}"}
            chunks.append(doc)

    return chunks


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
    # storing embeddings in the vector store
    # vectorstore = FAISS.from_texts(texts=chunks,
    #               embedding= embeddings)
    embeddings = select_embedding_model()
    vectorstore = Qdrant.from_documents(
        chunks,
        embeddings,
        location=":memory:",  # Local mode with in-memory storage only
        collection_name="my_documents",
    )
    return vectorstore


def get_conversation(vectorstore):
    llm = Ollama(base_url="http://localhost:11434", model="everythinglm:latest")
    # if st.session_state.llm_type == "Ollama":
    #     if st.session_state.ollama_model == "Mistral 7B":
    #         st.session_state.llm = Ollama(base_url = 'http://localhost:11434', model = 'mistral:instruct')
    #     if st.session_state.ollama_model == "Everythinglm 13B":
    #         st.session_state.llm = Ollama(base_url = 'http://localhost:11434', model = 'everythinglm:latest')
    #     if st.session_state.ollama_model == "Orca-Mini 3B":
    #         st.session_state.llm = Ollama(base_url = 'http://localhost:11434', model = 'orca-mini')
    if st.session_state.llm_type == "Ollama":
        if st.session_state.ollama_model == "Mistral 7B":
            llm = Ollama(base_url="http://localhost:11434", model="mistral:instruct")
        # if st.session_state.ollama_model == "Everythinglm 13B":
        # llm = Ollama(base_url = 'http://localhost:11434', model = 'everythinglm:latest')
        if st.session_state.ollama_model == "Orca-Mini 3B":
            llm = Ollama(base_url="http://localhost:11434", model="orca-mini")

    elif st.session_state.llm_type == "Google PaLM":
        google_api_key = st.secrets["palm_api_key"]
        llm = GooglePalm(
            google_api_key=google_api_key, model="chat-bison-001", temperatue=0.2
        )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    conversation = RetrievalQA.from_chain_type(
        llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_type="mmr"),
        memory=memory,
    )

    return conversation


def process_prompt():
    if prompt := st.chat_input("Ask a question about your documents"):
        st.session_state.chat_dialog_history.append({"role": "user", "content": prompt})
    # Display the prior chat messages
    for message in st.session_state.chat_dialog_history:
        with st.chat_message(name=message["role"]):
            st.write(message["content"])
    if st.session_state.chat_dialog_history[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # response = st.session_state.chat_engine.chat(prompt)
                response = st.session_state.conversation({"query": prompt})
                st.session_state.chat_history = response["chat_history"]
                st.write(response["result"])
                message = {"role": "assistant", "content": response["result"]}
                st.session_state.chat_dialog_history.append(message)


def process_document():
    with st.sidebar:
        load_models()
        st.info("⚠️ Ensure Docs are processed again on change of Model configurations")
        # for the documents
        if pdf_docs := st.file_uploader(
            "Upload the PDFs here:", accept_multiple_files=True
        ):
            if st.button("Process", type="primary", use_container_width=True):
                with st.spinner("Processing..."):
                    if (
                        st.session_state.disabled
                        or st.session_state.embedding_model_change_state
                    ):
                        text_chunks = get_text_chunks(pdf_docs)
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

    if st.session_state.disabled:
        st.write("🔒 Please upload and process your PDFs to unlock the question field.")

    else:
        process_prompt()


def main():
    load_ui()
    process_document()


if __name__ == "__main__":
    main()