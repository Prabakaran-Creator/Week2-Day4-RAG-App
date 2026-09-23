import os
import streamlit as st

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 PDF RAG Assistant")
st.write("Upload a PDF and ask questions based on its content.")


# --------------------------------------------------
# Check OpenAI API Key
# --------------------------------------------------

if not os.getenv("OPENAI_API_KEY"):
    st.error(
        "OPENAI_API_KEY is not set. "
        "Please configure your OpenAI API key as an environment variable."
    )
    st.stop()


# --------------------------------------------------
# Initialize Session State
# --------------------------------------------------

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "file_key" not in st.session_state:
    st.session_state.file_key = None


# --------------------------------------------------
# PDF Upload
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


# --------------------------------------------------
# Process PDF
# --------------------------------------------------

if uploaded_file is not None:

    current_file_key = f"{uploaded_file.name}_{uploaded_file.size}"

    # Process only when a new PDF is uploaded
    if st.session_state.file_key != current_file_key:

        with st.status("Processing PDF...", expanded=True) as status:

            # Step 1: Read PDF
            st.write("📖 Reading PDF...")

            reader = PdfReader(uploaded_file)

            pages_text = []

            for page in reader.pages:
                text = page.extract_text()

                if text:
                    pages_text.append(text)

            full_text = "\n\n".join(pages_text)

            # Check for readable text
            if not full_text.strip():
                status.update(
                    label="PDF processing failed",
                    state="error"
                )

                st.error(
                    "No readable text was found in this PDF. "
                    "Please upload a PDF containing selectable text."
                )

                st.stop()

            # Step 2: Split text
            st.write("✂️ Splitting document into chunks...")

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = text_splitter.create_documents(
                [full_text]
            )

            # Step 3: Create embeddings
            st.write("🧠 Creating OpenAI embeddings...")

            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small"
            )

            # Step 4: Create FAISS vector database
            st.write("🗄️ Building FAISS vector database...")

            vectorstore = FAISS.from_documents(
                chunks,
                embeddings
            )

            # Save vector store in session
            st.session_state.vectorstore = vectorstore
            st.session_state.file_key = current_file_key

            status.update(
                label="PDF is ready!",
                state="complete"
            )

        st.success(
            f"✅ {uploaded_file.name} is ready for questions."
        )


# --------------------------------------------------
# Question & Answer Section
# --------------------------------------------------

if st.session_state.vectorstore is not None:

    st.divider()

    st.subheader("💬 Ask a question about your PDF")

    question = st.text_input(
        "Enter your question:"
    )

    if question:

        with st.spinner("Searching the PDF and generating answer..."):

            # Create retriever
            retriever = st.session_state.vectorstore.as_retriever(
                search_kwargs={"k": 4}
            )

            # Retrieve relevant chunks
            relevant_docs = retriever.invoke(question)

            # Combine retrieved content
            context = "\n\n".join(
                doc.page_content
                for doc in relevant_docs
            )

            # Prompt
            prompt = ChatPromptTemplate.from_template(
                """
                You are a helpful PDF question-answering assistant.

                Answer the user's question using ONLY the
                information provided in the context below.

                If the answer cannot be found in the context,
                clearly say that the information was not found
                in the uploaded PDF.

                Context:
                {context}

                Question:
                {question}
                """
            )

            # OpenAI chat model
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0
            )

            # Generate answer
            messages = prompt.format_messages(
                context=context,
                question=question
            )

            response = llm.invoke(messages)

            # Display answer
            st.subheader("🤖 Answer")

            st.write(response.content)

            # Show retrieved sources
            with st.expander("📄 View retrieved PDF content"):

                for i, doc in enumerate(relevant_docs, start=1):

                    st.markdown(
                        f"**Retrieved Chunk {i}**"
                    )

                    st.write(doc.page_content)