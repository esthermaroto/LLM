import google.generativeai as genai

# Configurar Gemini
GOOGLE_API_KEY = "AIzaSyCQpipDouA0yE1MBUn2ySZeUjqKUTi5bCs"
genai.configure(api_key=GOOGLE_API_KEY)

# Listar modelos disponibles
print("Listando modelos disponibles de Gemini...")
for m in genai.list_models():
    print(f"Modelo: {m.name}")
    print(f"Descripción: {m.description}")
    print(f"Generación de contenido soportada: {m.supports_generate_content}")
    print("---") 