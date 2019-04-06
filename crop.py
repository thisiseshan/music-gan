from PIL import Image


for i in range(9):
    im = Image.open("/Users/eshan/Desktop/blues/blues0000" + str(i) + ".png")
    width, height = im.size
    left = (width - 558)/2
    top = (height - 544)/2
    right = (width + 558)/2
    bottom = (height + 544)/2
    image = im.crop((0, 0, right, bottom))
    width, height = image.size
    image.save("blues" + str(i) + ".png")
end
