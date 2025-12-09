"""
Módulo que contiene la aplicación del RAG
construido con Gemini en GCP usando Vertex AI.
Elaborado por: Edwin Caro | Andres Matallana | Santiago Zafra
"""

# Importar las librerías necesarias
from google import genai
from google.genai import types
import os
from config.constantes import Constantes as CONST

# Generar variable con la API key de Gemini en GCP
os.environ["GOOGLE_CLOUD_API_KEY"] = CONST.DATA["LLM_key"]

# Variable global para mantener el historial de la conversación
CHAT_HISTORY = []  # lista de dicts {"role": "user"/"model", "content": str}

# Configurar el llamado al RAG en GCP con Gemini
def generar_respuesta_rag(user_question: str) -> str:
    """
    Llama al modelo Gemini en Vertex AI usando un RAG corpus específico y retorna la respuesta.
    Además, mantiene un historial en memoria de la conversación (preguntas y respuestas) y lo
    envía como contexto en cada llamada.

    Args
    ----
    user_question : str
        Pregunta o instrucción que se le envía al modelo, en lenguaje natural.

    Returns
    -------
    str
        Texto generado por el modelo, ya enriquecido con el contexto del RAG corpus
        y el historial previo.
    """
    global CHAT_HISTORY

    # 1) Crear el cliente de Gemini en Vertex AI usando la API key del entorno
    client = genai.Client(
        vertexai=True,
        api_key=os.environ.get("GOOGLE_CLOUD_API_KEY"),
    )

    # 2) System prompt: define el rol del modelo
    si_text1 = CONST.DATA["Instrucciones_LLM"]

    # 3) Modelo a utilizar
    model_name = "gemini-2.5-flash"

    # 4) Construir contents a partir del historial + nueva pregunta
    contents = []

    # 4.1 Agregar historial previo (user/model)
    #     Opcional: limitar a los últimos N mensajes para no hacer el contexto gigante
    max_msgs = int(CONST.DATA["Cant_max_mensajes"])  # por ejemplo, últimos 10 mensajes (5 turnos)
    history_slice = CHAT_HISTORY[-max_msgs:]

    for msg in history_slice:
        contents.append(
            types.Content(
                role=msg["role"],
                parts=[types.Part.from_text(text=msg["content"])],
            )
        )

    # 4.2 Agregar la nueva pregunta del usuario
    contents.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_question)],
        )
    )

    # 5) Definir herramienta de recuperación RAG
    tools = [
        types.Tool(
            retrieval=types.Retrieval(
                vertex_rag_store=types.VertexRagStore(
                    rag_resources=[
                        types.VertexRagStoreRagResource(
                            rag_corpus=(
                                "projects/mintabot-maradoniano/locations/us-east1/"
                                "ragCorpora/6917529027641081856"
                            )
                        )
                    ],
                )
            )
        )
    ]

    # 6) Configuración de generación
    generate_content_config = types.GenerateContentConfig(
        temperature=0.0,
        top_p=1.0,
        max_output_tokens=64000,
        safety_settings=[
            types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="OFF",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="OFF",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="OFF",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="OFF",
            ),
        ],
        tools=tools,
        system_instruction=[types.Part.from_text(text=si_text1)],
        thinking_config=types.ThinkingConfig(thinking_budget=-1),
    )

    # 7) Llamada no streaming
    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=generate_content_config,
    )

    respuesta_txt = response.text

    # 8) Actualizar historial en memoria con el nuevo turno
    CHAT_HISTORY.append({"role": "user", "content": user_question})
    CHAT_HISTORY.append({"role": "model", "content": respuesta_txt})

    # 9) Retornar la respuesta
    return respuesta_txt
