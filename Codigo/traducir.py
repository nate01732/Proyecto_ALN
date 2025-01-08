from deep_translator import GoogleTranslator

def translate_file_keeping_format(input_file, output_file, source_lang='en', target_lang='es', max_length=5000):
    """
    Traduce el contenido de un archivo manteniendo el formato original.
    Divide los textos largos en fragmentos y respeta los saltos de línea.
    """
    try:
        translator = GoogleTranslator(source=source_lang, target=target_lang)

        with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
            for line in infile:
                # Si la línea es demasiado larga, dividirla en fragmentos
                chunks = split_text(line, max_length=max_length)
                translated_chunks = [translator.translate(chunk) for chunk in chunks]
                
                # Combinar los fragmentos traducidos y escribirlos en el archivo
                translated_line = ''.join(translated_chunks)
                outfile.write(translated_line + '\n')  # Mantener los saltos de línea
        
        print(f"Traducción completada. Contenido traducido guardado en {output_file}")
    
    except Exception as e:
        print(f"Ocurrió un error: {e}")

def split_text(text, max_length=5000):
    """
    Divide el texto en fragmentos de tamaño máximo especificado.
    """
    chunks = []
    while len(text) > max_length:
        # Encuentra el último espacio dentro del límite para evitar cortar palabras
        split_index = text.rfind(' ', 0, max_length)
        if split_index == -1:  # Si no hay espacio, corta directamente
            split_index = max_length
        chunks.append(text[:split_index])
        text = text[split_index:].strip()
    chunks.append(text)  # Añade el último fragmento
    return chunks

# Ejemplo de uso
input_file = 'resumen_maquinaria.txt'  # Nombre del archivo con el texto en inglés
output_file = 'translated.txt'  # Nombre del archivo donde guardarás el texto traducido

translate_file_keeping_format(input_file, output_file)
