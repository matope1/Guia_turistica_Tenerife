# 🧠 Guía Turística Inteligente con RAG y Function Calling

Este proyecto implementa un sistema de preguntas y respuestas basado en una guía turística utilizando una arquitectura RAG (Retrieval-Augmented Generation).  
El sistema permite realizar consultas sobre el contenido de una guía y también incluye una función externa para consultar el clima.

---

## 📌 Objetivo

El objetivo del proyecto es:

- Implementar un sistema RAG funcional.
- Usar embeddings para búsqueda semántica.
- Implementar recuperación con FAISS.
- Integrar generación con un modelo LLM (Gemini).
- Añadir memoria conversacional.
- Implementar una función externa (consulta del clima).
- Mostrar los chunks utilizados en cada respuesta para mayor transparencia.

---

## 🏗 Arquitectura del Sistema

El sistema sigue esta arquitectura:

1. **Carga y división del documento**
   - La guía turística se divide en fragmentos (chunks).
   - Esto permite búsquedas más precisas.

2. **Embeddings**
   - Cada chunk se transforma en un vector usando Gemini Embeddings.
   - Estos vectores representan el significado semántico del texto.

3. **Indexación**
   - Se utiliza FAISS para almacenar los vectores.
   - Permite búsqueda rápida por similitud.

4. **Recuperación**
   - Ante una pregunta del usuario, se genera su embedding.
   - Se buscan los 3 chunks más relevantes.

5. **Generación**
   - Se construye un prompt que incluye:
     - Pregunta del usuario
     - Contexto recuperado
   - Se genera la respuesta usando `gemini-2.5-flash-lite`.

6. **Memoria Conversacional**
   - Se almacena el historial de conversación.
   - Se limita a 5 turnos para evitar exceso de tokens(Modelo limitado al ser gratuito).

7. **Function Calling**
   - Se incluye una función simulada para consultar el clima.
   - Si el usuario pregunta por el tiempo, se ejecuta la función correspondiente.

8. **Transparencia**
   - En cada respuesta se muestran los chunks utilizados.

---

## 🛠 Tecnologías Utilizadas

- Python
- Google Generative AI (Gemini)
- FAISS (búsqueda vectorial)
- Streamlit (interfaz web)
- Logging para control de errores

---

## 🚀 Cómo ejecutar el proyecto

1. Crear entorno virtual:

```bash
python -m venv venv
```
```bash
.venv\Scripts\activate
```
2. Instalar dependencia:

```bash
pip install -r requirements.txt
```
3. Crear un archivo .env con tu API key de Gemini:
```bash
GOOGLE_API_KEY=tu_api_key_aqui
```
4. Ejecutar la aplicación Streamlit:
```bash
streamlit run app.py
```

