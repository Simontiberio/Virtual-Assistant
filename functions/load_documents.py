import os
import logging
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.output_parsers.rail_parser import GuardrailsOutputParser

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def cargar_documentos(ruta_archivo):
        
        """Carga y divide documentos en fragmentos para su procesamiento.

    Verifica si el archivo especificado existe y, si es así, utiliza PyMuPDFLoader
    para cargar el contenido. Luego, los documentos se dividen en fragmentos de texto 
    de tamaño controlado mediante un `RecursiveCharacterTextSplitter`.

    Args:
        ruta_archivo (str): Ruta al archivo PDF que se va a cargar.

    Returns:
        list: Lista de fragmentos de texto divididos.

    Raises:
        FileNotFoundError: Si el archivo no se encuentra en la ruta especificada. 
        
    """

    logging.info(f"Iniciando carga de documentos desde: {ruta_archivo}")

    if not os.path.exists(ruta_archivo):
        logging.error(f"El archivo {ruta_archivo} no existe.")
        raise FileNotFoundError(f"El archivo {ruta_archivo} no existe.")

    loader = PyMuPDFLoader(ruta_archivo)
    documentos = loader.load()
    logging.info(f"Documentos cargados exitosamente, total de documentos: {len(documentos)}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
    docs = text_splitter.split_documents(documentos)
    logging.info(f"Documentos divididos en {len(docs)} fragmentos para el procesamiento.")

    return docs