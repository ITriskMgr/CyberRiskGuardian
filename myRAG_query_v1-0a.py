# Version 3.1
# This version is streamlined for querying existing vectorstores only, removing file imports and vectorstore creation.

import logging
from langchain_ollama.llms import OllamaLLM
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains.retrieval_qa.base import RetrievalQA
import subprocess
import tkinter as tk
from tkinter import messagebox

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class QueryPipeline:
    def __init__(self, persist_directory):
        """
        Initializes the pipeline for querying an existing vectorstore.

        :param persist_directory: Directory path where the vectorstore is stored.
        """
        self.persist_directory = persist_directory
        self.model_name = None
        self.vectorstore = None
        self.qa_chain = None

    def list_ollama_models(self):
        """Fetches available Ollama models."""
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
        """Allows the user to select a model for querying."""
        models = self.list_ollama_models()
        if not models:
            logging.error("No models available. Please add models to Ollama.")
            exit()

        logging.info("Available models:")
        for idx, model in enumerate(models):
            logging.info(f"{idx + 1}. {model}")

        try:
            selected_idx = int(input("Select a model by number: ")) - 1
            if 0 <= selected_idx < len(models):
                self.model_name = models[selected_idx]
                logging.info(f"Selected model: {self.model_name}")
            else:
                logging.error("Invalid selection. Exiting.")
                exit()
        except ValueError:
            logging.error("Invalid input. Please enter a valid number.")
            exit()

    def setup_rag_pipeline(self):
        """Sets up the Retrieval-Augmented Generation pipeline."""
        logging.info("Setting up the Retrieval-Augmented Generation (RAG) pipeline...")
        try:
            # Ensure the embedding function is provided
            embedding_function = OllamaEmbeddings(model=self.model_name)
            self.vectorstore = Chroma(persist_directory=self.persist_directory, embedding_function=embedding_function)

            retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
            llm = OllamaLLM(model=self.model_name)
            self.qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
            logging.info("RAG pipeline is ready.")
        except Exception as e:
            logging.error(f"Error setting up RAG pipeline: {e}")
            exit()

    def query_pipeline(self, query):
        """Queries the RAG pipeline and returns the result."""
        logging.info(f"Querying the pipeline with: {query}")
        try:
            response = self.qa_chain.invoke({"query": query})
            result = response["result"]
            logging.info(f"Result: {result}")
            return result
        except Exception as e:
            logging.error(f"Error during query: {e}")
            return None

    def run_interactive_mode(self):
        """Launches a GUI for querying the RAG pipeline."""
        def on_query_submit():
            user_query = query_input.get()
            if user_query.strip():
                result = self.query_pipeline(user_query)
                if result:
                    messagebox.showinfo("Query Result", result)
                else:
                    messagebox.showerror("Error", "Failed to retrieve a response.")

        root = tk.Tk()
        root.title("RAG-enhanced LLM Query Tool")

        frame = tk.Frame(root, padx=10, pady=10)
        frame.pack(padx=10, pady=10)

        tk.Label(frame, text="Enter your query:").pack(anchor="w")
        query_input = tk.Entry(frame, width=50)
        query_input.pack(pady=5)

        submit_button = tk.Button(frame, text="Submit", command=on_query_submit)
        submit_button.pack(pady=5)

        root.mainloop()

if __name__ == "__main__":
    pipeline = QueryPipeline(persist_directory="./chroma_data")
    pipeline.select_model()
    pipeline.setup_rag_pipeline()

    logging.info("Launching interactive query interface...")
    pipeline.run_interactive_mode()
