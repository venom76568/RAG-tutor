from pathlib import Path
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io

class DocumentProcessor:
    @staticmethod
    def is_two_page_layout(image: Image.Image) -> bool:
        """
        Heuristic: Check if both halves of the image have significant OCR-detected text.
        """
        w, h = image.size
        left_img = image.crop((0, 0, w // 2, h))
        right_img = image.crop((w // 2, 0, w, h))

        left_text = pytesseract.image_to_string(left_img).strip()
        right_text = pytesseract.image_to_string(right_img).strip()

        return len(left_text) > 100 and len(right_text) > 100

    @staticmethod
    def load_pdf(file_path: Path) -> str:
        """
        Extracts text from a PDF, automatically detecting single vs double pages.
        Uses OCR when necessary. Leaves [IMAGE PAGE] if no text found.
        """
        file_path = Path(file_path)
        full_text = ""

        try:
            with fitz.open(file_path) as doc:
                for idx, page in enumerate(doc):
                    page_text = page.get_text().strip()

                    if not page_text:
                        # Fallback to OCR
                        pix = page.get_pixmap(dpi=300)
                        img = Image.open(io.BytesIO(pix.tobytes()))

                        # Check for double-page layout using OCR on halves
                        if DocumentProcessor.is_two_page_layout(img):
                            w, h = img.size
                            left_img = img.crop((0, 0, w // 2, h))
                            right_img = img.crop((w // 2, 0, w, h))

                            left_text = pytesseract.image_to_string(left_img).strip()
                            right_text = pytesseract.image_to_string(right_img).strip()
                            page_text = f"{left_text}\n{right_text}".strip()
                        else:
                            page_text = pytesseract.image_to_string(img).strip()

                        if not page_text:
                            page_text = "[IMAGE PAGE]"

                    full_text += f"\n--- Page {idx + 1} ---\n{page_text}\n"

            return full_text

        except Exception as e:
            raise RuntimeError(f"Failed to load PDF: {file_path.name} → {str(e)}")

