import os
import fitz  # PyMuPDF
from transformers import BlipProcessor, BlipForConditionalGeneration, LayoutLMTokenizer, LayoutLMForTokenClassification
from PIL import Image
import easyocr  # EasyOCR para reconocimiento de texto
import torch

# Configurar modelos BLIP
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Configurar lector de EasyOCR
reader = easyocr.Reader(['en', 'es'], gpu=False)  # Idiomas: inglés y español

# Configurar LayoutLM
layoutlm_tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
layoutlm_model = LayoutLMForTokenClassification.from_pretrained("microsoft/layoutlm-base-uncased")

# Rutas de archivos
pdf_path = r'C:\Users\herna\Downloads\KM-3175_OM MANUAL.pdf'
output_file = r"C:\Users\herna\Downloads\descripciones_imagenes1.txt"
temp_folder = r"C:\Users\herna\Downloads\temp_images"  # Carpeta temporal para imágenes

# Crear carpeta temporal si no existe
os.makedirs(temp_folder, exist_ok=True)

# Función para procesar una página
def process_page(page_number, doc, output_path):
    # Convertir la página a imagen
    page = doc.load_page(page_number)
    pix = page.get_pixmap(dpi=300)  # Ajustar DPI para mayor resolución
    img_path = os.path.join(temp_folder, f"temp_page_{page_number + 1}.png")
    pix.save(img_path)

    # Cargar y procesar la imagen
    with Image.open(img_path).convert('RGB') as img:
        # Generar descripción visual con BLIP
        inputs = processor(img, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=100, num_beams=5, repetition_penalty=1.5)
        visual_description = processor.decode(outputs[0], skip_special_tokens=True)

        # Extraer texto con EasyOCR
        ocr_results = reader.readtext(img_path, detail=1)  # Detalle=1 devuelve cajas de texto
        ocr_text = " ".join([text for _, text, _ in ocr_results])

        # Procesar con LayoutLM
        words = [text for _, text, _ in ocr_results]
        boxes = [box for box, _, _ in ocr_results]  # ajustar esta línea según la salida de EasyOCR
        width, height = img.size
        boxes = [[int(1000 * (box[0][0] / width)), int(1000 * (box[0][1] / height)),
                  int(1000 * (box[2][0] / width)), int(1000 * (box[2][1] / height))] for box in boxes]
        encoding = layoutlm_tokenizer(words, boxes=boxes, return_tensors="pt", truncation=True, padding="max_length")
        output = layoutlm_model(**encoding)
        token_predictions = output.logits.argmax(-1).squeeze().tolist()  # obtener el índice del máximo logit

        # Guardar descripciones en el archivo de salida
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(f"Página {page_number + 1}:\n")
            f.write(f"Descripción visual: {visual_description}\n")
            f.write(f"Texto extraído (OCR): {ocr_text.strip()}\n")
            f.write(f"Etiquetas LayoutLM: {[layoutlm_tokenizer.convert_ids_to_tokens(id) for id in token_predictions]}\n\n")

        print(f"Página {page_number + 1} procesada: Descripción y texto extraído.")

# Abrir el PDF con PyMuPDF
try:
    documento = fitz.open(pdf_path)
    # Procesar cada página
    for i in range(len(documento)):
        process_page(i, documento, output_file)
except Exception as e:
    print(f"Error al abrir el PDF: {e}")

# Limpiar archivos temporales
for file_name in os.listdir(temp_folder):
    os.remove(os.path.join(temp_folder, file_name))
os.rmdir(temp_folder)

# Cerrar el documento
documento.close()
print(f"Descripciones generadas y guardadas en: {output_file}")
