import streamlit as st
from langchain_cohere import ChatCohere
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import re

Cohere_API=st.secrets["API_KEYS"]["COHERE_API_KEY"]
st.set_page_config(
    page_title="Faculty BioGen",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        'About': 'Hello'
    }
)

with st.expander("Disclaimer"):
    st.markdown(
        """
        This application leverages AI to provide information and results. However, AI models may produce incorrect, outdated, or incomplete responses.
        Users are advised to cross-check and verify any critical information independently.
        """,
        unsafe_allow_html=True,
    )


TEXT_FILE_PATH = "GraphicEraFaculty.txt"

def text_file_to_text(text_file):
    try:
        with open(text_file, 'r', encoding='utf-8') as file:
            text = file.read()
        return text
    except FileNotFoundError:
        st.error(f"File {text_file} not found.")
        return None
    except Exception as e:
        st.error(f"An error occurred while reading the file: {e}")
        return None

def text_splitter(raw_text):
    try:
        chunks = re.split(r'\n(?=\d+\.)', raw_text)
        cleaned_chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        
        return cleaned_chunks
    except Exception as e:
        st.error(f"An error occurred while splitting the text: {e}")
        return None

def get_vector_store(text_chunks):
    try:
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        vectorstore = FAISS.from_texts(text_chunks, embedding=embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"An error occurred while creating the vector store: {e}")
        return None

def format_docs(docs):
    try:
        return "\n\n".join(doc.page_content for doc in docs)
    except Exception as e:
        st.error(f"An error occurred while formatting the documents: {e}")
        return None

def generate_answer(question, retriever):
    try:
        cohere_llm=ChatCohere(model="command-r7b-12-2024", cohere_api_key=Cohere_API, temperature=0.3)
        prompt_template = """Using the provided context, answer the question as accurately and comprehensively as possible:
                                - If the question asks about a specific faculty member, summarize their profile based on the available information.
                                - If the question asks about a certain piece of information (e.g., research areas, achievements, roles, or designations like HOD), extract and present the relevant details from the context.
                                - If the question asks about faculty members in a specific role, identify and present the corresponding faculty member and their designation from the context.
                                - If the requested information is not found in the context, respond with: "Sorry, the requested information is not available in the provided context."

                            Context: 
                            {context}
                            
                            Question: 
                            {question}

                            Answer:"""

        prompt = PromptTemplate.from_template(template=prompt_template)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | cohere_llm
            | StrOutputParser()
        )

        return rag_chain.invoke(question)
    except Exception as e:
        st.error(f"An error occurred while generating the answer: {e}")
        return None

def main():
    st.header("Faculty BioGen")
    st.write("Hello, welcome! Feel free to ask anything about any faculty of the Computer Science department.")

    question = st.text_input("Ask a question:")

    if st.button("Ask Questions"):
        with st.spinner("Please have patience for a moment I am searching the available information :)"):
            raw_text = text_file_to_text(TEXT_FILE_PATH)
            if raw_text is None:
                return

            text_chunks = text_splitter(raw_text)
            if text_chunks is None:
                return

            vectorstore = get_vector_store(text_chunks)
            if vectorstore is None:
                return

            retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

            if question:
                answer = generate_answer(question, retriever)
                if answer:
                    st.write(answer)
                else:
                    st.write("Sorry, but I don't have that information.")
            else:
                st.warning("Please enter a question.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
