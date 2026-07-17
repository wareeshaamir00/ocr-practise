from PIL import Image
im = Image.open("images\image.webp")
im.show()
im.save("output\im.png")