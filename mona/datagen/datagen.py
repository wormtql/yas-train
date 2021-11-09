import random
import os

from PIL import Image, ImageFont, ImageDraw

from mona.text.artifact_name import random_artifact_name
from mona.text.stat import random_sub_stat, random_main_stat_name, random_main_stat_value
from mona.text.characters import random_equip
from mona.config import config
from mona.datagen.pre_process import pre_process


fonts = [ImageFont.truetype("./assets/genshin.ttf", i) for i in range(15, 40)]


def random_level():
    return "+" + str(random.randint(0, 20))


flag_ac = 0
def random_artifact_count():
    global flag_ac
    temp = f"圣遗物 {flag_ac}/1500"
    flag_ac += 1
    if flag_ac >= 1501:
        flag_ac = 0
    return temp


random_funcs = [random_artifact_name, random_main_stat_name, random_main_stat_value,
                random_sub_stat, random_level, random_equip, random_artifact_count]
random_weights = [0.1, 0.05, 0.15,
                  0.64, 0.02, 0.02, 0.02]


def rand_color_1():
    r = random.randint(150, 255)
    g = random.randint(150, 255)
    b = random.randint(150, 255)

    # temp = random.choice(background_colors)
    # r = min(255, temp[0] + random.randint(-20, 20))
    # g = min(255, temp[1] + random.randint(-20, 20))
    # b = min(255, temp[2] + random.randint(-20, 20))

    return r, g, b


def rand_color_2():
    r = random.randint(0, 100)
    g = random.randint(0, 100)
    b = random.randint(0, 100)

    # temp = random.choice(font_colors)
    # r = min(255, temp[0] + random.randint(-20, 20))
    # g = min(255, temp[1] + random.randint(-20, 20))
    # b = min(255, temp[2] + random.randint(-20, 20))

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

    img = Image.new("RGB", (500, 50), color1)
    # img = Image.new("RGB", (config["train_width"], config["height"]), color1)
    draw = ImageDraw.Draw(img)
    x = random.randint(0, 20)
    y = random.randint(0, 5)

    text = random_text()

    draw.text((x, y), text, color2, font=random.choice(fonts))

    img = pre_process(img)
    # img = resize_to_32(img)
    # print(img.size)
    return img, text
