import streamlit as st
import os
import logging
import numpy as np
import faiss
import PyPDF2
import google.genai as genai
import re
from dotenv import load_dotenv


load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("No se encontró GOOGLE_API_KEY")
    st.stop()

client = genai.Client(api_key=GOOGLE_API_KEY)
MODEL_NAME = "gemini-2.5-flash-lite"

logging.basicConfig(level=logging.INFO)



# CARGA Y RAG 

@st.cache_resource
def build_index():

    # 1. Cargar PDF
    text = ""
    with open("TENERIFE.pdf", "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"

    # 2. Chunking
    chunks = []
    chunk_size = 500
    overlap = 50
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    # 3. Embeddings
    embeddings = []
    for chunk in chunks:
        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=chunk
        )
        emb = np.array(response.embeddings[0].values, dtype="float32")
        embeddings.append(emb)

    # 4. FAISS
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)

    for emb in embeddings:
        index.add(emb.reshape(1, -1))

    return index, chunks


index, chunks = build_index()



# FUNCIONES

def get_embedding(text):
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return np.array(response.embeddings[0].values, dtype="float32")


def search_chunks(query, top_k=3):
    query_emb = get_embedding(query).reshape(1, -1)
    distances, indices = index.search(query_emb, top_k)

    context = ""
    for i in indices[0]:
        context += f"\n[Chunk {i}]\n{chunks[i]}\n"

    return context


def get_weather(fecha: str):
    logging.info(f"Llamada a get_weather con fecha: {fecha}")

    weather_data = {
        "2024-05-01": "Soleado, 26°C",
        "2024-05-02": "Parcialmente nublado, 24°C",
        "2024-05-03": "Lluvia ligera, 22°C",
    }

    return weather_data.get(fecha, "No hay datos disponibles para esa fecha.")


def chat_rag(user_input):

    # Detectar clima
    if "clima" in user_input.lower() or "tiempo" in user_input.lower():
        match = re.search(r"\d{4}-\d{2}-\d{2}", user_input)
        if match:
            fecha = match.group()
            return f"Predicción del tiempo para {fecha}: {get_weather(fecha)}"

    # Recuperación
    context = search_chunks(user_input)

    # Historial
    history_text = ""
    for msg in st.session_state.chat_history[-5:]:
        history_text += f"Usuario: {msg['user']}\nAsistente: {msg['assistant']}\n"

    prompt = f"""
Eres un asistente turístico experto en Tenerife.

Historial:
{history_text}

Contexto:
{context}

Pregunta:
{user_input}

Responde citando la fuente como:
Fuente: Chunk X
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config={
            "temperature": 0.7,
            "max_output_tokens": 500,
        }
    )

    return (response.text or "").strip()




# INTERFAZ STREAMLIT

st.title("🌋 Asistente Turístico de Tenerife")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Mostrar historial
for msg in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(msg["user"])
    with st.chat_message("assistant"):
        st.write(msg["assistant"])

# Input del usuario
user_input = st.chat_input("Haz tu pregunta...")

if user_input:
    with st.chat_message("user"):
        st.write(user_input)

    response = chat_rag(user_input)

    with st.chat_message("assistant"):
        st.write(response)

    st.session_state.chat_history.append({
        "user": user_input,
        "assistant": response
    })



# PREGUNTAS DE EJEMPLO
st.sidebar.header("Preguntas de ejemplo")
example_questions = [
    # General
    "Háblame del Teide",
    "Qué playas puedo visitar",
    "Recomiéndame restaurantes en Santa Cruz",
    "Qué lugares son buenos para ver el atardecer",
    "Qué actividades se pueden hacer en Puerto de la Cruz",
    
    # Clima function call
    "Qué tiempo hará el 2024-05-01",
    "Predicción del clima para el 2024-12-01",
    
    # multiturno
    "Me llamo Guillermo, ¿podrías acordarte de llamarme por mi nombre en cada respuesta, por favor?"
]


# Mostrar cada pregunta como botón
for q in example_questions:
    if st.sidebar.button(q):
        # Copiar la pregunta al input del chat
        user_input = q
        # Simular envío
        with st.chat_message("user"):
            st.write(user_input)

        response = chat_rag(user_input)

        with st.chat_message("assistant"):
            st.write(response)

        st.session_state.chat_history.append({
            "user": user_input,
            "assistant": response
        })
