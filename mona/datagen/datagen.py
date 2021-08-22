import random
import os

from PIL import Image, ImageFont, ImageDraw

from mona.text.artifact_name import random_artifact_name
from mona.text.stat import random_stat, random_value

print(__file__)
font32 = ImageFont.truetype("./assets/genshin.ttf", 20)


random_funcs = [random_artifact_name, random_stat, random_value]
random_weights = [0.2, 0.4, 0.4]


def rand_color_1():
    r = random.randint(150, 255)
    g = random.randint(150, 255)
    b = random.randint(150, 255)
    return r, g, b


def rand_color_2():
    r = random.randint(0, 75)
    g = random.randint(0, 75)
    b = random.randint(0, 75)
    return r, g, b


def random_text():
    func = random.choices(
        population=random_funcs,
        weights=random_weights,
        k=1
    )
    return func[0]()


def generate_image():
    color1 = rand_color_1()
    color2 = rand_color_2()

    if random.random() < 0.5:
        color1, color2 = color2, color1

    img = Image.new("RGB", (224, 32), color1)
    draw = ImageDraw.Draw(img)
    x = random.randint(0, 10)
    y = random.randint(0, 5)

    text = random_text()

    draw.text((x, y), text, color2, font=font32)

    return img, text
