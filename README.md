# Sistema RAG con Gemini

Este proyecto implementa un sistema de Recuperación Aumentada de Generación (RAG) utilizando Google Gemini como modelo base. El sistema está diseñado para extraer información precisa de páginas web específicas y responder consultas basándose únicamente en el contenido de estas páginas.

## Características

- Extracción de contenido de páginas web
- Procesamiento y almacenamiento de embeddings usando ChromaDB
- Generación de respuestas precisas usando Gemini
- Sistema de recuperación semántica
- Respuestas basadas únicamente en el contenido de las páginas web proporcionadas

## Requisitos

- Python 3.8 o superior
- API key de Google Gemini

## Instalación

1. Clona este repositorio
2. Instala las dependencias:
```bash
pip install -r requirements.txt
```

3. Crea un archivo `.env` en la raíz del proyecto y añade tu API key de Google:
```
GOOGLE_API_KEY=tu_api_key_aquí
```

## Uso

1. Modifica el archivo `rag_system.py` para incluir las URLs de las páginas web que deseas procesar:
```python
urls = [
    "https://ejemplo1.com",
    "https://ejemplo2.com"
]
```

2. Ejecuta el sistema:
```bash
python rag_system.py
```

## Estructura del Proyecto

- `rag_system.py`: Implementación principal del sistema RAG
- `requirements.txt`: Dependencias del proyecto
- `.env`: Archivo de configuración para variables de entorno

## Notas Importantes

- El sistema solo responderá basándose en la información extraída de las páginas web proporcionadas
- Se requiere una API key válida de Google Gemini
- El sistema utiliza ChromaDB para el almacenamiento de embeddings
- Las respuestas se generan con una temperatura baja (0.1) para maximizar la precisión 