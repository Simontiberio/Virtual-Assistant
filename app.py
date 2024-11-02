import streamlit as st
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from functions.vectors_create import crear_vectorstore
from functions.load_documents import cargar_documentos
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# Códigos de escape ANSI para colores
AZUL = "\033[94m"
VERDE = "\033[92m"
RESET = "\033[0m"
def iniciar_chat(ruta_archivo):
    llm = Ollama(model="llama3")
    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


    vectorstore = Chroma(embedding_function=embed_model,
                                       persist_directory="chroma_db_dir_Governace",
                                       collection_name="data_Governance_2024")
    total_rows = len(vectorstore.get()['ids'])
    if total_rows == 0:
        docs = cargar_documentos(ruta_archivo)
        vectorstore = crear_vectorstore(docs)
    retriever = vectorstore.as_retriever(search_kwargs={'k': 4})

    custom_prompt_template = """ Tu eres un asistente de la asignatura del Gobierno del dato, en el ambito educativo. Responderas consultas
    administrativas y operativas respecto al curso.
    Usa la siguiente información para responder a la pregunta del usuario.
    Si no sabes la respuesta, simplemente di que no lo sabes, no intentes inventar una respuesta.

    Contexto: {context}
    Pregunta: {question}

    Solo devuelve la respuesta útil a continuación y nada más y responde siempre en español
    Respuesta útil:
    """
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    print("¡Hola Bienvenid@! Soy Sara, ¿En que puedo ayudarte? Escribe 'salir' para terminar.")
    while True:
        pregunta = input(f"{AZUL}Tú:{RESET} ")
        if pregunta.lower() == 'salir':
            print("¡Hasta luego!")
            break

        respuesta = qa.invoke({"query": pregunta})
        metadata = []
        for _ in respuesta['source_documents']:
            metadata.append(('page: '+str(_.metadata['page']), _.metadata['file_path']))
        print(f"{VERDE}Asistente:{RESET}", respuesta['result'], '\n', metadata)

# Interfaz con Streamlit
def main():
    st.title("Asistente Virtual de Documentos")
    
    ruta_archivo = "documents/Data Governance 2024.docx"
    qa = iniciar_chat(ruta_archivo)
    
    st.write("¡Bienvenido al chat!")
    st.write("Escribe 'salir' para terminar.")

    pregunta = st.text_input("Escribe tu pregunta aquí:")
    
    if pregunta:
        if pregunta.lower() == 'salir':
            st.write("¡Hasta luego!")
        else:
            respuesta = qa.invoke({"query": pregunta})
            metadata = []
            for doc in respuesta['source_documents']:
                metadata.append(('page: ' + str(doc.metadata['page']), doc.metadata['file_path']))
            
            st.write(f"Asistente: {respuesta['result']}")
            st.write("Documentos fuente:", metadata)

if __name__ == "__main__":
    ruta_archivo = "documents\Data_Governance_2024.docx"

    iniciar_chat(ruta_archivo)