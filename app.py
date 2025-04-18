import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
    CSVLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import tempfile

# Load environment variables
load_dotenv()

# Initialize session state
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None


def get_file_loader(file_path, file_type):
    """Return appropriate loader based on file type"""
    loaders = {
        "pdf": PyPDFLoader,
        "docx": Docx2txtLoader,
        "txt": TextLoader,
        "md": UnstructuredMarkdownLoader,
        "html": UnstructuredHTMLLoader,
        "doc": UnstructuredWordDocumentLoader,
        "xlsx": UnstructuredExcelLoader,
        "pptx": UnstructuredPowerPointLoader,
        "csv": CSVLoader,
    }
    return loaders.get(file_type.lower(), TextLoader)(file_path)


def process_documents(documents):
    """Process documents and create vector store"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    """Create conversation chain"""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", temperature=0, convert_system_message_to_human=True
    )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )
    return conversation_chain


def main():
    st.set_page_config(page_title="RAG Chat Demo", page_icon="🤖")
    st.title("RAG Chat Demo 🤖")

    # Sidebar for file upload
    with st.sidebar:
        st.subheader("Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload your documents",
            type=[
                "pdf",
                "docx",
                "txt",
                "md",
                "html",
                "doc",
                "xlsx",
                "pptx",
                "csv",
            ],
            accept_multiple_files=True,
        )

        if uploaded_files:
            with st.spinner("Processing documents..."):
                all_documents = []
                for uploaded_file in uploaded_files:
                    # Create a temporary file
                    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name

                    # Get file extension
                    file_extension = uploaded_file.name.split(".")[-1]

                    try:
                        # Load document
                        loader = get_file_loader(tmp_file_path, file_extension)
                        documents = loader.load()
                        all_documents.extend(documents)
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                    finally:
                        # Clean up temporary file
                        os.unlink(tmp_file_path)

                if all_documents:
                    # Process documents and create vector store
                    st.session_state.vectorstore = process_documents(all_documents)
                    st.session_state.conversation = get_conversation_chain(
                        st.session_state.vectorstore
                    )
                    st.success("Documents processed successfully!")

    # Main chat interface
    if st.session_state.conversation is None:
        st.info("Please upload documents to start chatting.")
    else:
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask a question about your documents"):
            with st.chat_message("user"):
                st.write(prompt)

            # Get response
            with st.spinner("Thinking..."):
                response = st.session_state.conversation({"question": prompt})
                answer = response["answer"]

                with st.chat_message("assistant"):
                    st.write(answer)

                # Update chat history
                st.session_state.chat_history.append(
                    {"role": "user", "content": prompt}
                )
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": answer}
                )


if __name__ == "__main__":
    main()
