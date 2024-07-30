import json
import random

from mona.text.text_generator import TextGenerator


# echoes_name = [
#     "呼咻咻",
#     "咔嚓嚓",
#     "阿嗞嗞",
#     "呜咔咔",
#     "叮咚咚",
#     "咕咕河豚",
#     "啾啾河豚",
#     "遁地鼠",
#     "绿熔蜥（稚形）",
#     "碎獠猪",
#     ""
# ]

with open("assets/wuthering_waves/MultiText.json", encoding="utf-8") as f:
    text_map_json = json.load(f)
with open("assets/wuthering_waves/phantomitem.json", encoding="utf-8") as f:
    phantom_item = json.load(f)

text_map = {}
for item in text_map_json:
    text_map[item["Id"]] = item["Content"]

name_set = set()
for item in phantom_item:
    monster_name = item["MonsterName"]
    name = text_map[monster_name]
    name_set.add(name)

names = list(name_set)


class WWEchoesNameGenerator(TextGenerator):
    def __init__(self):
        super(WWEchoesNameGenerator, self).__init__("Wuthering Waves Echoes Name")

    def generate_text(self):
        return random.choice(names)

    def get_lexicon(self):
        ret = set()
        for name in names:
            for c in name:
                ret.add(c)
        return ret
