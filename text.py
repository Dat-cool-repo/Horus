import pytesseract
from PIL import Image

# Specify Tesseract executable path (if not in PATH)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Open an image file
image_path = "test.jpg"
text = pytesseract.image_to_string(Image.open(image_path), lang='eng')

# Print the extracted text
print(text)
