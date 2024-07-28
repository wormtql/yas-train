import random

from mona.text.text_generator import TextGenerator


class GenshinArtifactCountGenerator(TextGenerator):
    def __init__(self):
        super(GenshinArtifactCountGenerator, self).__init__("Genshin Artifact Count")

    def generate_text(self):
        flag_ac = random.randint(0, 1800)
        return f"圣遗物 {flag_ac}/1800"

    def get_lexicon(self):
        ret = set()
        for c in " 0123456789.+%,圣遗物":
            ret.add(c)
        return ret


class GenshinArtifactLevelGenerator(TextGenerator):
    def __init__(self):
        super(GenshinArtifactLevelGenerator, self).__init__("Genshin Artifact Level")

    def generate_text(self):
        return "+" + str(random.randint(0, 20))

    def get_lexicon(self):
        ret = set()
        for c in "0123456789+":
            ret.add(c)
        return ret
