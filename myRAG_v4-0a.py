import os
import random
import subprocess
import logging
from tqdm import tqdm
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.schema import Document
from langchain_chroma import Chroma
import tkinter as tk
from tkinter import messagebox
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class DocumentProcessorPipeline:
    def __init__(self, input_directory, persist_directory, num_test_files):
        self.input_directory = input_directory
        self.persist_directory = persist_directory
        self.num_test_files = num_test_files
        self.documents = []
        self.text_chunks = []
        self.vectorstore = None
        self.selected_model = None
        self.qa_chain = None

    def ensure_ollama_path(self):
        logging.info("Checking if `ollama` is available in PATH...")
        ollama_path = subprocess.run(["which", "ollama"], capture_output=True, text=True).stdout.strip()
        if not ollama_path:
            logging.warning("`ollama` not found. Adding `/usr/local/bin` to PATH...")
            os.environ['PATH'] += ':/usr/local/bin'
            ollama_path = subprocess.run(["which", "ollama"], capture_output=True, text=True).stdout.strip()
            if not ollama_path:
                logging.error("Failed to locate `ollama`. Ensure it is installed in `/usr/local/bin`.")
                exit()
        logging.info(f"`ollama` found at: {ollama_path}")

    def fetch_files(self):
        try:
            logging.info("Fetching files from input directory...")
            files = os.listdir(self.input_directory)
            supported_extensions = ['.pdf', '.txt']
            document_files = [f for f in files if os.path.splitext(f)[1].lower() in supported_extensions]

            if len(document_files) < self.num_test_files:
                logging.error("Not enough files to select. Exiting.")
                exit()

            self.documents = random.sample(document_files, self.num_test_files)
            logging.info(f"Selected files: {self.documents}")
        except Exception as e:
            logging.error(f"Error while fetching files: {e}")
            exit()

    def load_documents(self):
        logging.info("Loading documents...")
        loaded_documents = []
        for file in tqdm(self.documents, desc="Loading files"):
            file_path = os.path.join(self.input_directory, file)
            try:
                if file.lower().endswith('.pdf'):
                    reader = PdfReader(file_path)
                    content = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
                    if content.strip():
                        loaded_documents.append({"content": content, "metadata": {"source": file_path}})
                    else:
                        logging.warning(f"Empty content extracted from {file}. Skipping.")
                elif file.lower().endswith('.txt'):
                    with open(file_path, 'r') as f:
                        content = f.read()
                        if content.strip():
                            loaded_documents.append({"content": content, "metadata": {"source": file_path}})
                        else:
                            logging.warning(f"Empty content in {file}. Skipping.")
                else:
                    logging.warning(f"Unsupported file type: {file}")
            except Exception as e:
                logging.error(f"Failed to load {file}: {e}")
        self.documents = loaded_documents

    def split_documents(self):
        logging.info("Splitting documents into smaller chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        formatted_documents = [Document(page_content=doc["content"], metadata=doc["metadata"]) for doc in
                               self.documents]
        self.text_chunks = text_splitter.split_documents(formatted_documents)
        logging.info(f"Generated {len(self.text_chunks)} text chunks.")

    def generate_bibliography(self):
        logging.info("Generating bibliography in APA format...")
        bibliography_entries = []
        for doc in self.documents:
            source = doc["metadata"]["source"]
            file_name = os.path.basename(source)
            file_extension = os.path.splitext(file_name)[1]
            title = os.path.splitext(file_name)[0].replace("_", " ").title()

            if file_extension == ".pdf":
                entry = f"{title}. (n.d.). Retrieved from {source}"
            elif file_extension == ".txt":
                entry = f"{title}. (n.d.). Retrieved from {source}"
            else:
                entry = f"{title}. (n.d.)."

            bibliography_entries.append(entry)

        bibliography_text = "\n\n".join(bibliography_entries)
        bibliography_path = os.path.join(self.persist_directory, "bibliography.txt")

        try:
            os.makedirs(self.persist_directory, exist_ok=True)  # Ensure the directory exists
            with open(bibliography_path, "w") as f:
                f.write(bibliography_text)
            logging.info(f"Bibliography saved to {bibliography_path}")
        except Exception as e:
            logging.error(f"Failed to save bibliography: {e}")

    def list_ollama_models(self):
        logging.info("Fetching available models from Ollama...")
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            if result.returncode != 0:
                logging.error(f"Error listing models: {result.stderr.strip()}")
                return []
            models_output = result.stdout.strip().split("\n")[1:]
            return [line.split()[0] for line in models_output if line.strip()]
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            return []

    def select_model(self):
        logging.info("Selecting a model for embedding generation...")
        available_models = self.list_ollama_models()

        if not available_models:
            logging.error("No models available. Exiting.")
            exit()

        logging.info("Available models:")
        for idx, model in enumerate(available_models):
            logging.info(f"{idx + 1}. {model}")

        try:
            selected_idx = int(input("Select a model by number: ")) - 1
            if 0 <= selected_idx < len(available_models):
                self.selected_model = available_models[selected_idx]
                logging.info(f"Selected model: {self.selected_model}")
            else:
                logging.error("Invalid selection. Exiting.")
                exit()
        except ValueError:
            logging.error("Invalid input. Please enter a valid number.")
            exit()

    def initialize_vectorstore(self):
        logging.info("Initializing vectorstore and generating embeddings...")
        try:
            self.embeddings = OllamaEmbeddings(model=self.selected_model)
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )

            for idx, text_chunk in enumerate(tqdm(self.text_chunks, desc="Adding to vectorstore")):
                chunk_id = f"chunk_{idx}"
                chunk_embedding = self.embeddings.embed_documents([text_chunk.page_content])[0]

                self.vectorstore._collection.upsert(
                    ids=[chunk_id],
                    embeddings=[chunk_embedding],
                    metadatas=[text_chunk.metadata],
                    documents=[text_chunk.page_content]
                )

            logging.info(f"Vectorstore persisted at '{self.persist_directory}'.")
        except Exception as e:
            logging.error(f"Error initializing vectorstore: {e}")
            exit()

    def setup_rag_pipeline(self):
        logging.info("Setting up the Retrieval-Augmented Generation (RAG) pipeline...")
        try:
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
            llm = OllamaLLM(model=self.selected_model)
            self.qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
            logging.info("RAG pipeline is ready.")
        except Exception as e:
            logging.error(f"Error setting up RAG pipeline: {e}")
            exit()

    def query_pipeline(self, query):
        logging.info(f"Querying the pipeline with: {query}")
        try:
            response = self.qa_chain.invoke({"query": query})
            result = response.get("result")
            logging.info(f"Query result: {result}")
            return result
        except Exception as e:
            logging.error(f"Error during query: {e}")
            return None

    def run_interactive_mode(self):
        def on_query_submit():
            user_query = query_input.get()
            if user_query.strip():
                result = self.query_pipeline(user_query)
                if result:
                    messagebox.showinfo("Query Result", result)
                else:
                    messagebox.showerror("Error", "Failed to retrieve a response.")

        root = tk.Tk()
        root.title("RAG-enhanced LLM query interface")

        frame = tk.Frame(root, padx=10, pady=10)
        frame.pack(padx=10, pady=10)

        tk.Label(frame, text="Enter your query:").pack(anchor="w")
        query_input = tk.Entry(frame, width=50)
        query_input.pack(pady=5)

        submit_button = tk.Button(frame, text="Submit your query", command=on_query_submit)
        submit_button.pack(pady=5)

        root.mainloop()


if __name__ == "__main__":
    pipeline = DocumentProcessorPipeline(
        input_directory="/Users/maleger/My_documents",
        persist_directory="./chroma_data",
        num_test_files=5
    )
    pipeline.ensure_ollama_path()
    pipeline.fetch_files()
    pipeline.load_documents()
    pipeline.split_documents()
    pipeline.select_model()
    pipeline.generate_bibliography()
    pipeline.initialize_vectorstore()
    pipeline.setup_rag_pipeline()

    logging.info("Launching interactive query interface...")
    pipeline.run_interactive_mode()
