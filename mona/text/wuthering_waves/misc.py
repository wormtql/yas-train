import random

from mona.text.text_generator import TextGenerator


class WWEchosLevelGenerator(TextGenerator):
    def __init__(self, **kwargs):
        super(WWEchosLevelGenerator, self).__init__("Wuthering Waves Echo Level")

    def generate_text(self):
        return "+" + str(random.randint(0, 25))

    def get_lexicon(self):
        ret = set()
        for c in "0123456789+":
            ret.add(c)
        return ret


class WWEchoesCountGenerator(TextGenerator):
    def __init__(self, max_count = 2000):
        super(WWEchoesCountGenerator, self).__init__("Wutering Waves Echo Count")
        self.max_count = max_count

    def generate_text(self):
        return "声骸 " + f"{random.randint(0, self.max_count)}/{self.max_count}"

    def get_lexicon(self):
        ret = set()
        for c in " 0123456789声骸/":
            ret.add(c)
        return ret
