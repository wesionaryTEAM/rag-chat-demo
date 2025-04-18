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

# Load environment variables from .env file
load_dotenv()

# Initialize session state variables to maintain state across Streamlit reruns
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None


def get_file_loader(file_path: str, file_type: str):
    """
    Returns the appropriate document loader based on the file type.
    
    Args:
        file_path (str): Path to the file to be loaded
        file_type (str): Extension of the file (e.g., 'pdf', 'docx', 'txt')
        
    Returns:
        DocumentLoader: An instance of the appropriate document loader class
    """
    # Dictionary mapping file extensions to their corresponding loader classes
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
    """
    Processes documents by splitting them into chunks and creating a vector store.
    
    Args:
        documents (List[Document]): List of documents to be processed
        
    Returns:
        FAISS: A vector store containing the document embeddings
    """
    # Split documents into chunks with overlap for better context preservation
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Maximum size of each chunk
        chunk_overlap=200,  # Number of characters to overlap between chunks
        length_function=len  # Function to calculate chunk length
    )
    chunks = text_splitter.split_documents(documents)

    # Create embeddings using Google's Generative AI model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # Create and return a FAISS vector store from the document chunks
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    """
    Creates a conversation chain with memory for maintaining chat history.
    
    Args:
        vectorstore (FAISS): Vector store containing document embeddings
        
    Returns:
        ConversationalRetrievalChain: A chain that can handle conversations with document retrieval
    """
    # Initialize the language model with specific parameters
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",  # Model to use for generating responses
        temperature=0,  # Controls randomness in responses (0 = deterministic)
    )
    
    # Initialize memory to store chat history
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # Create and return a conversation chain with document retrieval capabilities
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,  # Language model for generating responses
        retriever=vectorstore.as_retriever(),  # Document retriever
        memory=memory  # Memory for maintaining conversation history
    )
    return conversation_chain


def main():
    """
    Main function that sets up the Streamlit interface and handles the chat application logic.
    """
    # Configure Streamlit page settings
    st.set_page_config(page_title="RAG Chat Demo", page_icon="🤖")
    st.title("RAG Chat Demo 🤖")

    # Sidebar for file upload
    with st.sidebar:
        st.subheader("Upload Documents")
        # File uploader that accepts multiple file types
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
                    # Create a temporary file to store the uploaded content
                    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name

                    # Get file extension for determining the appropriate loader
                    file_extension = uploaded_file.name.split(".")[-1]

                    try:
                        # Load and process the document
                        loader = get_file_loader(tmp_file_path, file_extension)
                        documents = loader.load()
                        all_documents.extend(documents)
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                    finally:
                        # Clean up the temporary file
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

            # Get response from the conversation chain
            with st.spinner("Thinking..."):
                response = st.session_state.conversation({"question": prompt})
                answer = response["answer"]

                with st.chat_message("assistant"):
                    st.write(answer)

                # Update chat history with the new exchange
                st.session_state.chat_history.append(
                    {"role": "user", "content": prompt}
                )
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": answer}
                )


if __name__ == "__main__":
    main()
