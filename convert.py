from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.document_converter import DocumentConverter

source = "book.pdf"
converter = DocumentConverter()
result = converter.convert(source)

if result.status == ConversionStatus.SUCCESS:
    doc = result.document
    text = doc.export_to_text()
    with open("book.txt", "w", encoding="utf-8") as f:
        f.write(text)
else:
    print("Conversion failed:", result.status, getattr(result, "error", None))
