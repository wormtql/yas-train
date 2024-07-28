import random

from mona.text.text_generator import TextGenerator


class StarrailRelicLevelGenerator(TextGenerator):
    def __init__(self):
        super(StarrailRelicLevelGenerator, self).__init__("Starrail Relic Level")

    def generate_text(self):
        return "+" + str(random.randint(0, 20))

    def get_lexicon(self):
        ret = set()
        for c in "0123456789+":
            ret.add(c)
        return ret


class StarrailRelicCountGenerator(TextGenerator):
    def __init__(self):
        super(StarrailRelicCountGenerator, self).__init__("Starrail Relic Count")

    def generate_text(self):
        # Random here, for online learning
        flag_ac = random.randint(0, 2000)
        return f"遗器数量{flag_ac}/2000"

    def get_lexicon(self):
        ret = set()
        for c in " '0123456789.,/遗器数量":
            ret.add(c)
        return ret
