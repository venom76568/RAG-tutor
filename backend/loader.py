from pathlib import Path
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io

class DocumentProcessor:
    @staticmethod
    def load_pdf(file_path: Path) -> str:
        """
        Extract text from a PDF file.
        Falls back to OCR using Tesseract if no extractable text is found.
        """
        file_path = Path(file_path)
        text = ""

        try:
            with fitz.open(file_path) as doc:
                for page in doc:
                    text += page.get_text()

            if not text.strip():
                # No extractable text found — use OCR
                images = []
                with fitz.open(file_path) as doc:
                    for page in doc:
                        pix = page.get_pixmap()
                        img = Image.open(io.BytesIO(pix.tobytes()))
                        images.append(img)

                text = "\n".join(pytesseract.image_to_string(img) for img in images)

            return text

        except Exception as e:
            raise RuntimeError(f"Failed to load PDF: {file_path.name} → {str(e)}")


# Example usage
if __name__ == "__main__":
    processor = DocumentProcessor()
    extracted_text = processor.load_pdf("huffman_codes.pdf")
    print(extracted_text)
