import os
import json
from typing import List, Dict
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM, OllamaEmbeddings
import time
import concurrent.futures
from queue import Queue
import threading

# Cargar variables de entorno
load_dotenv()

# Definir rutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")
EMBEDDINGS_FILE = os.path.join(EMBEDDINGS_DIR, "embeddings.json")

# Crear directorio de embeddings si no existe
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

class WebRAGSystem:
    def __init__(self):
        """Inicializa el sistema RAG con múltiples modelos de Ollama."""
        print("Inicializando modelos de Ollama...")
        
        # Modelos para procesar archivos en paralelo
        self.worker_models = [
            OllamaLLM(model="mistral", temperature=0.1),
            OllamaLLM(model="mistral", temperature=0.1),
            OllamaLLM(model="mistral", temperature=0.1)
        ]
        
        # Modelo final para integrar y generar respuestas
        self.final_model = OllamaLLM(
            model="mistral",
            temperature=0.7
        )
        
        # Inicializar embeddings con Ollama
        self.embeddings = OllamaEmbeddings(
            model="mistral"
        )
        
        print("Inicializando ChromaDB...")
        # Inicializar ChromaDB
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.create_collection(name="web_content")
        self.cache_collection = self.chroma_client.create_collection(name="query_cache")
        self.context_collection = self.chroma_client.create_collection(name="retrieved_context")
        
        # Inicializar el splitter de texto
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=30
        )
        
        # Cola para resultados procesados
        self.results_queue = Queue()
        
        # Cargar embeddings existentes
        self.load_embeddings()
        
        print("Sistema inicializado correctamente.")

    def load_embeddings(self):
        """Carga los embeddings desde el archivo JSON."""
        try:
            if os.path.exists(EMBEDDINGS_FILE):
                with open(EMBEDDINGS_FILE, 'r', encoding='utf-8') as f:
                    embeddings_data = json.load(f)
                    print(f"Embeddings cargados: {len(embeddings_data)} documentos")
                    return embeddings_data
            return {}
        except Exception as e:
            print(f"Error al cargar embeddings: {str(e)}")
            return {}

    def save_embeddings(self, embeddings_data: Dict):
        """Guarda los embeddings en el archivo JSON."""
        try:
            with open(EMBEDDINGS_FILE, 'w', encoding='utf-8') as f:
                json.dump(embeddings_data, f, ensure_ascii=False, indent=2)
            print(f"Embeddings guardados: {len(embeddings_data)} documentos")
        except Exception as e:
            print(f"Error al guardar embeddings: {str(e)}")

    def extract_web_content(self, file_path: str) -> str:
        """Extrae el contenido de un archivo HTML local."""
        try:
            print(f"Leyendo archivo: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            print("Procesando HTML...")
            soup = BeautifulSoup(content, 'html.parser')
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            print(f"Contenido extraído: {len(text)} caracteres")
            return text
        except Exception as e:
            print(f"Error al extraer contenido de {file_path}: {str(e)}")
            return ""

    def process_with_model(self, content: str, model_index: int, file_name: str) -> str:
        """Procesa el contenido con un modelo específico."""
        try:
            print(f"Modelo {model_index} procesando {file_name}...")
            prompt = f"""¡Hola! Necesito que me ayudes a extraer la información más importante sobre FicZone 2025.
            Me interesa especialmente saber sobre:
            • Fechas exactas del evento
            • Dónde se celebrará, ubicación exacta 
            • Qué eventos principales habrá
            • Los horarios más importantes
            • Quiénes son los invitados confirmados

            Aquí está el texto para analizar:
            {content}

            Por favor, organiza la información de forma clara usando viñetas. ¡Gracias!"""
            
            response = self.worker_models[model_index].invoke(prompt)
            print(f"Modelo {model_index} completó el procesamiento de {file_name}")
            return response
        except Exception as e:
            print(f"Error en modelo {model_index} procesando {file_name}: {str(e)}")
            return ""

    def process_file_parallel(self, file_path: str, model_index: int):
        """Procesa un archivo usando un modelo específico y guarda el resultado en la cola."""
        content = self.extract_web_content(file_path)
        if content:
            processed_content = self.process_with_model(content, model_index, os.path.basename(file_path))
            self.results_queue.put((file_path, processed_content))

    def process_and_store_content(self, files: List[str]):
        """Procesa y almacena el contenido de los archivos en paralelo."""
        print("\nIniciando procesamiento paralelo...")
        
        # Cargar embeddings existentes
        embeddings_data = self.load_embeddings()
        
        # Verificar qué archivos ya están procesados
        existing_sources = set(embeddings_data.keys())
        
        # Filtrar archivos que necesitan ser procesados
        files_to_process = [f for f in files if os.path.basename(f) not in existing_sources]
        
        if not files_to_process:
            print("Todos los archivos ya están procesados.")
            return
            
        print(f"Procesando {len(files_to_process)} archivos nuevos...")
        
        # Procesar archivos en paralelo
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.worker_models)) as executor:
            futures = []
            for i, file in enumerate(files_to_process):
                model_index = i % len(self.worker_models)
                futures.append(executor.submit(self.process_file_parallel, file, model_index))
            
            # Esperar a que todos los archivos sean procesados
            concurrent.futures.wait(futures)
        
        # Procesar resultados de la cola
        while not self.results_queue.empty():
            file_path, processed_content = self.results_queue.get()
            file_name = os.path.basename(file_path)
            
            # Dividir el contenido procesado en chunks
            print(f"\nDividiendo contenido de {file_name} en chunks...")
            chunks = self.text_splitter.split_text(processed_content)
            print(f"Generados {len(chunks)} chunks")
            
            # Generar embeddings y almacenar
            print("Generando embeddings...")
            file_embeddings = []
            for i, chunk in enumerate(chunks):
                try:
                    embedding = self.embeddings.embed_query(chunk)
                    file_embeddings.append({
                        "chunk_id": i,
                        "text": chunk,
                        "embedding": embedding
                    })
                except Exception as e:
                    print(f"Error generando embedding para chunk {i} de {file_name}: {str(e)}")
                    continue
            
            # Guardar embeddings del archivo
            embeddings_data[file_name] = file_embeddings
            
            # Almacenar en ChromaDB
            print("Almacenando en ChromaDB...")
            for embedding_data in file_embeddings:
                try:
                    self.collection.add(
                        ids=[f"{file_name}_{embedding_data['chunk_id']}"],
                        embeddings=[embedding_data['embedding']],
                        documents=[embedding_data['text']],
                        metadatas=[{"source": file_name, "chunk_id": embedding_data['chunk_id']}]
                    )
                except Exception as e:
                    print(f"Error almacenando en ChromaDB: {str(e)}")
                    continue
            
            # Guardar embeddings actualizados
            self.save_embeddings(embeddings_data)

    def check_cache(self, question: str, similarity_threshold: float = 0.85) -> str:
        """Verifica si existe una respuesta similar en la caché."""
        try:
            # Generar embedding de la pregunta
            query_embedding = self.embeddings.embed_query(question)
            
            # Buscar en la caché
            results = self.cache_collection.query(
                query_embeddings=[query_embedding],
                n_results=1
            )
            
            if results['distances'][0] and results['distances'][0][0] <= (1 - similarity_threshold):
                print("Respuesta encontrada en caché")
                return results['documents'][0][0]
            
            return None
        except Exception as e:
            print(f"Error al verificar caché: {str(e)}")
            return None

    def add_to_cache(self, question: str, answer: str):
        """Añade una pregunta y su respuesta a la caché."""
        try:
            # Generar embedding de la pregunta
            query_embedding = self.embeddings.embed_query(question)
            
            # Añadir a la caché
            self.cache_collection.add(
                ids=[f"q_{len(self.cache_collection.get()['ids'])}"],
                embeddings=[query_embedding],
                documents=[answer],
                metadatas=[{"question": question}]
            )
            print("Respuesta añadida a la caché")
        except Exception as e:
            print(f"Error al añadir a caché: {str(e)}")

    def check_context_cache(self, question: str, similarity_threshold: float = 0.85) -> str:
        """Verifica si existe contexto relevante en la caché de contexto."""
        try:
            # Generar embedding de la pregunta
            query_embedding = self.embeddings.embed_query(question)
            
            # Buscar en la caché de contexto
            results = self.context_collection.query(
                query_embeddings=[query_embedding],
                n_results=3  # Buscar los 3 contextos más relevantes
            )
            
            if results['distances'][0] and any(dist <= (1 - similarity_threshold) for dist in results['distances'][0]):
                print("Contexto relevante encontrado en caché")
                # Combinar los contextos relevantes
                relevant_contexts = []
                for i, dist in enumerate(results['distances'][0]):
                    if dist <= (1 - similarity_threshold):
                        relevant_contexts.append(results['documents'][0][i])
                return "\n".join(relevant_contexts)
            
            return None
        except Exception as e:
            print(f"Error al verificar caché de contexto: {str(e)}")
            return None

    def add_to_context_cache(self, question: str, context: str):
        """Añade un contexto recuperado a la caché."""
        try:
            # Generar embedding de la pregunta
            query_embedding = self.embeddings.embed_query(question)
            
            # Añadir a la caché de contexto
            self.context_collection.add(
                ids=[f"ctx_{len(self.context_collection.get()['ids'])}"],
                embeddings=[query_embedding],
                documents=[context],
                metadatas=[{"question": question}]
            )
            print("Contexto añadido a la caché")
        except Exception as e:
            print(f"Error al añadir contexto a caché: {str(e)}")

    def query(self, question: str, k: int = 5) -> str:
        """Realiza una consulta al sistema RAG usando el modelo final para respuestas integradas."""
        print("\nVerificando caché de respuestas...")
        cached_response = self.check_cache(question)
        if cached_response:
            return cached_response
        
        print("\nVerificando caché de contexto...")
        cached_context = self.check_context_cache(question)
        context = None
        
        if cached_context:
            print("Usando contexto de caché")
            context = cached_context
        else:
            print("Buscando información en documentos originales...")
            query_embedding = self.embeddings.embed_query(question)
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k
            )
            
            if results['documents'][0]:
                context = "\n".join(results['documents'][0])
                # Almacenar el contexto recuperado
                self.add_to_context_cache(question, context)
            else:
                print("No se encontró información en el contexto almacenado. Buscando en archivos de texto...")
                file_paths = [os.path.join('data', 'processed', url) for url in self.get_available_files()]
                relevant_files = self.filter_relevant_files(file_paths, question)
                self.process_and_store_content(relevant_files)
                
                # Intentar la búsqueda nuevamente después de procesar los archivos
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=k
                )
                if results['documents'][0]:
                    context = "\n".join(results['documents'][0])
                    self.add_to_context_cache(question, context)
                else:
                    return "Lo siento, no he podido encontrar información relevante para tu pregunta."
        
        print("Información relevante encontrada")
        print("\nContexto recuperado:")
        print(context)
        
        print("Generando respuesta integrada...")
        
        # Verificar si la pregunta es sobre horarios
        if any(word in question.lower() for word in ['horario', 'horarios', 'hora', 'cuándo', 'cuando', 'abre', 'cierra']):
            prompt = f"""¡Hola! Voy a ayudarte con tu pregunta sobre los horarios de FicZone 2025.
            Para darte la mejor respuesta, seguiré estas pautas:
            
            1. Resumen General:
            • Especificar claramente las fechas del evento
            • Horario de apertura y cierre para cada día
            • Si hay diferentes horarios para diferentes zonas, especificarlo
            
            2. Detalles por Día:
            • Organizar la información por día
            • Para cada día, listar los eventos principales con sus horarios
            • Incluir cualquier cambio de horario especial
            
            3. Formato:
            • Usar viñetas para mejor legibilidad
            • Separar claramente la información por días
            • Destacar los horarios más importantes
            
            Con esta información:
            {context}

            Tu pregunta es: {question}

            Por favor, responde en español de forma amigable, siguiendo exactamente esta estructura:
            1. Primero el resumen general con fechas y horarios de apertura/cierre
            2. Luego los detalles organizados por día
            3. Finalmente cualquier información adicional relevante sobre horarios"""
        else:
            prompt = f"""¡Hola! Voy a ayudarte con tu pregunta sobre FicZone 2025.
            Para darte la mejor respuesta, seguiré estas pautas:
            • Responderé de forma clara y directa
            • Usaré un tono amigable y cercano
            • Si no tengo la información, te lo diré con sinceridad
            • Organizaré la información en viñetas para que sea fácil de leer
            • Me centraré en lo más importante

            Con esta información:
            {context}

            Tu pregunta es: {question}

            Por favor, responde en español de forma amigable:"""
        
        response = self.final_model.invoke(prompt)
        
        # Añadir la respuesta a la caché
        self.add_to_cache(question, response)
        
        return response

    def get_available_files(self) -> List[str]:
        """Obtiene la lista de archivos disponibles en el directorio processed."""
        try:
            # Obtener todos los archivos .txt del directorio processed
            files = []
            for file in os.listdir(PROCESSED_DIR):
                if file.endswith('.txt') and file != 'urls.txt':
                    files.append(file)
            return files
        except Exception as e:
            print(f"Error al leer archivos disponibles: {str(e)}")
            return []

    def filter_relevant_files(self, files: List[str], question: str) -> List[str]:
        """Filtra los archivos que probablemente contengan información relevante para la pregunta."""
        print("\nFiltrando archivos relevantes...")
        relevant_files = []
        
        # Palabras clave para diferentes tipos de preguntas
        keywords = {
            "invitados": ["invitado", "invitados", "ponente", "ponentes", "artista", "artistas"],
            "fechas": ["fecha", "fechas", "cuándo", "día", "días", "horario", "horarios"],
            "eventos": ["evento", "eventos", "actividad", "actividades", "programa"],
            "ubicación": ["ubicación", "lugar", "dirección", "localización"]
        }
        
        # Determinar el tipo de pregunta
        question_lower = question.lower()
        question_type = None
        for key, words in keywords.items():
            if any(word in question_lower for word in words):
                question_type = key
                break
        
        if not question_type:
            print("No se pudo determinar el tipo de pregunta, procesando todos los archivos")
            return files
        
        # Filtrar archivos basado en el tipo de pregunta
        for file in files:
            file_lower = file.lower()
            if question_type == "invitados" and "invitados" in file_lower:
                relevant_files.append(file)
            elif question_type == "fechas" and any(word in file_lower for word in ["horarios", "plano"]):
                relevant_files.append(file)
            elif question_type == "eventos" and any(word in file_lower for word in ["ficzone", "meeple", "granada"]):
                relevant_files.append(file)
            elif question_type == "ubicación" and any(word in file_lower for word in ["granada", "ubicación"]):
                relevant_files.append(file)
        
        print(f"Archivos relevantes encontrados: {len(relevant_files)}")
        for file in relevant_files:
            print(f"- {file}")
        
        return relevant_files if relevant_files else files

def main():
    # Inicializar el sistema
    rag = WebRAGSystem()
    
    # Obtener lista de archivos disponibles
    urls = rag.get_available_files()
    
    print(f"\nArchivos disponibles: {len(urls)}")
    for url in urls:
        print(f"- {url}")
    
    # Procesar todos los archivos al inicio
    print("\nProcesando archivos iniciales...")
    file_paths = [os.path.join('data', 'processed', url) for url in urls]
    rag.process_and_store_content(file_paths)
    
    # Bucle interactivo para preguntas
    print("\nSistema listo para preguntas. Escribe 'salir' para terminar.")
    print("Comandos disponibles:")
    print("- 'salir': Terminar el programa")
    print("- 'actualizar': Procesar archivos nuevos")
    
    while True:
        try:
            # Obtener pregunta del usuario
            question = input("\nTu pregunta: ").strip()
            
            # Verificar si el usuario quiere salir
            if question.lower() in ['salir', 'exit', 'quit']:
                print("\n¡Hasta luego!")
                break
            
            # Verificar si el usuario quiere actualizar
            if question.lower() == 'actualizar':
                print("\nProcesando archivos nuevos...")
                file_paths = [os.path.join('data', 'processed', url) for url in urls]
                rag.process_and_store_content(file_paths)
                print("Archivos procesados correctamente.")
                continue
            
            if not question:
                print("Por favor, escribe una pregunta.")
                continue
            
            # Realizar la consulta
            print(f"\nProcesando pregunta: {question}")
            response = rag.query(question)
            
            print("\nRespuesta del sistema:")
            print(response)
            
        except KeyboardInterrupt:
            print("\n\n¡Hasta luego!")
            break
        except Exception as e:
            print(f"\nError al procesar la pregunta: {str(e)}")
            continue

if __name__ == "__main__":
    main() 