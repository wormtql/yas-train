import random

from PIL import Image, ImageFont, ImageDraw

from mona.datagen.pre_process import pre_process


def rand_color_1():
    r = random.randint(150, 255)
    g = random.randint(150, 255)
    b = random.randint(150, 255)

    return r, g, b


def rand_color_2():
    r = random.randint(0, 100)
    g = random.randint(0, 100)
    b = random.randint(0, 100)

    return r, g, b


class DataGen:
    def __init__(self, config, fonts, lexicon):
        self.config = config
        self.fonts = fonts
        self.lexicon = lexicon

    def generate_image(self):
        color1 = rand_color_1()
        color2 = rand_color_2()

        img = Image.new("RGB", (1200, 120), color1)
        draw = ImageDraw.Draw(img)
        x = random.randint(0, 20)
        y = random.randint(0, 5)

        text = self.lexicon.generate_text()

        draw.text((x, y), text, color2, font=random.choice(self.fonts))

        # Random binarization thr to mimic various rendering
        thr = random.uniform(0.5, 0.6)
        img = pre_process(img, thr)
        return img, text

    def generate_image_sample(self):
        color1 = rand_color_1()
        color2 = rand_color_2()

        img = Image.new("RGB", (1200, 120), color1)
        draw = ImageDraw.Draw(img)
        x = random.randint(0, 20)
        y = random.randint(0, 5)

        text = self.lexicon.generate_text()

        # This would disable anit-aliasing
        # draw.fontmode = "1"

        draw.text((20, 5), "雷素%暴岩1,7.", color2, font=ImageFont.truetype("./assets/genshin.ttf", 80))

        img_processed = pre_process(img)
        return img, img_processed
