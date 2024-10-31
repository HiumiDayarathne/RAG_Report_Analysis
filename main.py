import streamlit as st
import fitz  # PyMuPDF
import io
from langchain_openai import OpenAI
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Streamlit app
st.title("ReportInsight")

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# File uploader
uploaded_file = st.file_uploader("Upload a document", type=["pdf"])

if uploaded_file is not None:
    # Load PDF content with PyMuPDF
    pdf_content = []
    try:
        with fitz.open(stream=io.BytesIO(uploaded_file.read()), filetype="pdf") as pdf_doc:
            for page_num in range(len(pdf_doc)):
                page = pdf_doc[page_num]
                text = page.get_text()
                pdf_content.append({"page": page_num, "text": text})

        st.write("Document content loaded successfully.")
    except Exception as e:
        st.error(f"Error loading Document: {str(e)}")

    # Split the content into chunks with page numbers
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs_with_metadata = []
    for item in pdf_content:
        splits = text_splitter.split_text(item["text"])
        for split in splits:
            docs_with_metadata.append({"text": split, "metadata": {"page": item["page"]}})

    # Create vector store with metadata
    vectorstore = InMemoryVectorStore.from_texts(
        texts=[doc["text"] for doc in docs_with_metadata],
        embedding=OpenAIEmbeddings(),
        metadatas=[doc["metadata"] for doc in docs_with_metadata]
    )

    retriever = vectorstore.as_retriever()

    # Define the LLM using OpenAI class
    llm = OpenAI(api_key=api_key, temperature=0.7, max_tokens=500)

    # Define system prompt
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question concisely. Indicate the page numbers where "
        "the information was found, if possible. Keep the answer to four sentences.  If you don't know the answer, say that you "
        "don't know."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # Input question
    question = st.text_input("Ask a question about the document:")

    if st.button("Get Answer"):
        if question:
            # Invoke the RAG chain
            results = rag_chain.invoke({"input": question})
            st.write("Answer:")

            # Collect answers with page metadata
            answer = results["context"][0].page_content
            pages = {doc.metadata["page"] for doc in results["context"]}

            # Store the question and answer in chat history
            st.session_state.chat_history.append({"question": question, "answer": answer})

            # Display answer with page numbers
            st.markdown(f"**{answer}**")
            st.write("#### Source Pages:")
            st.write(", ".join([f"Page {page}" for page in sorted(pages)]))
        else:
            st.warning("Please enter a question.")

    # Display chat history
    if st.session_state.chat_history:
        st.write("### Chat History")
        for chat in st.session_state.chat_history:
            st.markdown(f"**Q:** {chat['question']}")
            st.markdown(f"**A:** {chat['answer']}")
