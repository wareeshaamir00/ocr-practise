
import pytesseract
from main3 import preprocess

image = preprocess()

text = pytesseract.image_to_string(image)
print(text)
with open("output/text.txt", "w", encoding="utf-8") as f:
    f.write(text)