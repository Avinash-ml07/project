from pdfminer.high_level import extract_text
from unstructured.partition.auto import partition
import fitz  # PyMuPDF

class PDFProcessor:
    def __init__(self):
        pass
    
    def extract_with_pdfminer(self, pdf_path):
        """Extract text using PDFMiner for text-based PDFs"""
        try:
            text = extract_text(pdf_path)
            return text
        except Exception as e:
            print(f"PDFMiner extraction failed: {e}")
            return None
    
    def extract_with_unstructured(self, pdf_path):
        """Extract structured elements using Unstructured library"""
        elements = partition(pdf_path)
        
        # Organize elements by type
        structured_content = {
            'titles': [],
            'text': [],
            'tables': [],
            'lists': []
        }
        
        for element in elements:
            if element.category == "Title":
                structured_content['titles'].append(str(element))
            elif element.category == "NarrativeText":
                structured_content['text'].append(str(element))
            elif element.category == "Table":
                structured_content['tables'].append(str(element))
            elif element.category == "ListItem":
                structured_content['lists'].append(str(element))
        
        return structured_content
