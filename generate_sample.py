from PIL import ImageFont

from mona.config import config
from mona.text import get_lexicon
from mona.datagen.datagen import DataGen


lexicon = get_lexicon(config["model_type"])

# 4k分辨率最大对应84号字，900p分辨率最小对应18号字
if config["model_type"] == "Genshin":
    fonts = [ImageFont.truetype("./assets/genshin.ttf", i) for i in range(15, 90)]
elif config["model_type"] == "StarRail":
    fonts = [ImageFont.truetype("./assets/starrail.ttf", i) for i in range(15, 90)]
elif config["model_type"] == "WutheringWaves":
    fonts = [ImageFont.truetype("./assets/wuthering_waves/ARFangXinShuH7GBK-HV.ttf", i) for i in range(15, 90)]
datagen = DataGen(config, fonts, lexicon)


for i in range(10):
    im, label = datagen.generate_image()
    im.save(f"samples/{label}.png")
