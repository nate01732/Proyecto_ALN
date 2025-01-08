import PyPDF2
import nltk
import requests
from nltk.corpus import stopwords
from transformers import pipeline, AutoTokenizer
from collections import Counter
import enchant  

nltk.download('punkt')
nltk.download('stopwords')

# Diccionario de inglés para validación de palabras
diccionario_ingles = enchant.Dict("en_US")

# Función para extraer texto de un PDF
def extraer_texto_de_pdf(ruta_pdf):
    try:
        with open(ruta_pdf, 'rb') as archivo:
            lector = PyPDF2.PdfReader(archivo)
            texto = ""
            for pagina in lector.pages:
                if pagina.extract_text():
                    texto += pagina.extract_text()
            return texto
    except Exception as e:
        print(f"Error al leer el PDF: {e}")
        return None

# Función para dividir el texto en fragmentos de tokens
def dividir_texto_en_fragmentos_de_tokens(texto, tokenizador, max_tokens=1000):
    palabras = nltk.word_tokenize(texto)
    fragmentos = []
    fragmento_actual = []
    longitud_actual = 0

    for palabra in palabras:
        longitud_token = len(tokenizador.tokenize(palabra))
        if longitud_actual + longitud_token > max_tokens:
            fragmentos.append(" ".join(fragmento_actual))
            fragmento_actual = []
            longitud_actual = 0
        fragmento_actual.append(palabra)
        longitud_actual += longitud_token

    if fragmento_actual:
        fragmentos.append(" ".join(fragmento_actual))
    
    return fragmentos

# Función para resumir texto
def resumir_texto(texto, longitud_min=80, longitud_max=200):
    resumidor = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", framework="pt")
    tokenizador = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
    
    fragmentos = dividir_texto_en_fragmentos_de_tokens(texto, tokenizador, max_tokens=1000)
    resumenes = []
    
    for fragmento in fragmentos:
        try:
            longitud_fragmento = len(tokenizador.tokenize(fragmento))
            longitud_min, longitud_max = ajustar_longitudes(longitud_min, longitud_max, longitud_fragmento)
            if longitud_fragmento <= 1024:
                resumen = resumidor(fragmento, max_length=longitud_max, min_length=longitud_min, do_sample=False)
                resumenes.append(resumen[0]['summary_text'])
        except Exception as e:
            print(f"Error al resumir un fragmento: {e}")
    
    return " ".join(resumenes)

# Función para ajustar las longitudes del resumen
def ajustar_longitudes(longitud_min, longitud_max, longitud_fragmento):
    if longitud_max >= longitud_fragmento:
        longitud_max = longitud_fragmento - 1
    if longitud_min >= longitud_max:
        longitud_min = longitud_max - 1
    return longitud_min, longitud_max

# Función para obtener definiciones contextuales de términos
def obtener_definicion_contextual(palabra, contexto):
    try:
        respuesta = requests.get(f"https://api.dictionaryapi.dev/api/v2/entries/en/{palabra}")
        if respuesta.status_code == 200:
            datos = respuesta.json()
            for significado in datos[0].get('meanings', []):
                for definicion in significado.get('definitions', []):
                    if palabra in contexto:
                        return definicion['definition']
        return "Definición no disponible. Consulte el manual."
    except Exception as e:
        print(f"Error al obtener la definición de '{palabra}': {e}")
        return "Definición no disponible. Consulte el manual."

# Función mejorada para generar un glosario
def generar_glosario(texto, num_terminos=20):
    palabras_comunes = set(stopwords.words('english'))
    palabras = nltk.word_tokenize(texto)
    palabras = [palabra.lower() for palabra in palabras if palabra.isalpha() and palabra.lower() not in palabras_comunes and diccionario_ingles.check(palabra)]
    contexto = texto.lower()
    palabras_no_comunes = Counter(palabras).most_common(num_terminos)
    glosario = {palabra: obtener_definicion_contextual(palabra, contexto) for palabra, _ in palabras_no_comunes}
    return glosario

# Función para estructurar y verificar secciones del resumen
def estructurar_resumen(resumen, texto_completo):
    oraciones = nltk.tokenize.sent_tokenize(resumen)
    num_oraciones = len(oraciones)

    recomendaciones = " ".join(oraciones[:num_oraciones // 4])
    precauciones = " ".join(oraciones[num_oraciones // 4:num_oraciones // 2])
    instrucciones_texto = " ".join(oraciones[num_oraciones // 2:num_oraciones * 3 // 4])
    
    # Verificar si existen palabras clave para las secciones
    if "caution" not in precauciones.lower():
        precauciones += "\nInformación adicional sobre precauciones: " + extraer_informacion_relacionada(texto_completo, "precaución")
    
    pasos_instrucciones = nltk.tokenize.sent_tokenize(instrucciones_texto)
    instrucciones = "\n".join([f"{i+1}. {paso}" for i, paso in enumerate(pasos_instrucciones)])
    
    referencias = "Para más información, consulte el manual completo."

    return {
        "Recomendaciones": recomendaciones,
        "Precauciones": precauciones,
        "Instrucciones": instrucciones,
        "Referencias": referencias
    }

# Función para extraer información relacionada del texto
def extraer_informacion_relacionada(texto, palabra_clave):
    oraciones = nltk.tokenize.sent_tokenize(texto)
    informacion_relacionada = [oracion for oracion in oraciones if palabra_clave in oracion.lower()]
    return " ".join(informacion_relacionada)

# Función para guardar los resultados en un archivo TXT
def guardar_en_txt(glosario, resumen_estructurado, ruta_salida="resumen_maquinaria.txt"):
    with open(ruta_salida, "w", encoding="utf-8") as archivo:
        archivo.write("GLOSARIO:\n")
        for termino, definicion in glosario.items():
            archivo.write(f"- {termino}: {definicion}\n")
        
        archivo.write("\nRESUMEN ESTRUCTURADO:\n")
        for seccion, contenido in resumen_estructurado.items():
            archivo.write(f"\n{seccion.upper()}:\n{contenido}\n")
    print(f"Resumen guardado en {ruta_salida}")

# Función principal para procesar el manual
def procesar_manual(ruta_pdf):
    print("Extrayendo texto del PDF...")
    texto = extraer_texto_de_pdf(ruta_pdf)
    if not texto:
        return None
    
    print("Generando glosario...")
    glosario = generar_glosario(texto)
    
    print("Resumiendo texto...")
    resumen = resumir_texto(texto)
    
    print("Estructurando resumen...")
    resumen_estructurado = estructurar_resumen(resumen, texto)
    
    print("Guardando resultados...")
    guardar_en_txt(glosario, resumen_estructurado)

# Ejemplo de uso
if __name__ == "__main__":
    ruta_pdf = "manual.pdf"  
    procesar_manual(ruta_pdf)
