import os
import logging
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.output_parsers.rail_parser import GuardrailsOutputParser
from load_documents import cargar_documentos


# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def crear_vectorstore(docs):

        """Crea y persiste un almacén vectorial (VectorStore) para consulta eficiente de documentos.

    Esta función utiliza HuggingFaceEmbeddings para transformar los fragmentos de texto en 
    vectores y luego los guarda en un almacén vectorial de Chroma, persistiendo la información 
    para futuras consultas.

    Args:
        docs (list): Lista de fragmentos de texto procesados.

    Returns:
        Chroma: Instancia del almacén vectorial creado.

    Raises:
        Exception: Si ocurre un error durante la creación o persistencia del VectorStore.
    """

    logging.info("Iniciando la creación del VectorStore.")
    logging.info("Cargando el modelo de embeddings de Hugging Face.")

    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    logging.info(f"Modelo de embeddings '{embed_model.model_name}' cargado correctamente.")

    try:
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embed_model,
            persist_directory="chroma_db_dir_Governace",
            collection_name="Data_Governance_2024"
        )
        logging.info("VectorStore creado y persistido en 'chroma_db_dir_Governace'.")
    except Exception as e:
        logging.error("Error al crear el VectorStore.", exc_info=True)
        raise e

    return vectorstore
