from mona.text.genshin import *
from mona.text.starrail import *
from mona.text.common import *
from mona.text.lexicon import Lexicon


def construct_genshin_lexicon():
    random_artifact_name = GenshinArtifactTextGenerator()
    random_main_stat_name = GenshinMainStatNameGenerator()
    random_main_stat_value = GenshinMainStatValueGenerator()
    random_sub_stat = GenshinSubStatGenerator()
    random_level = GenshinArtifactLevelGenerator()
    random_equip = GenshinCharacterEquipTextGenerator()
    random_artifact_count = GenshinArtifactCountGenerator()
    random_number_generator = RandomNumberTextGenerator()

    weighted_generator = WeightedTextGenerator()
    weighted_generator.add_entry(0.1, random_artifact_name)
    weighted_generator.add_entry(0.05, random_main_stat_name)
    weighted_generator.add_entry(0.15, random_main_stat_value)
    weighted_generator.add_entry(0.64, random_sub_stat)
    weighted_generator.add_entry(0.02, random_level)
    weighted_generator.add_entry(0.02, random_equip)
    weighted_generator.add_entry(0.1, random_artifact_count)
    weighted_generator.add_entry(0.2, random_number_generator)

    return weighted_generator


def construct_starrail_lexicon():
    random_relic_name = StarrailRelicNameGenerator()
    random_main_stat_name = StarrailMainStatNameGenerator()
    random_main_stat_value = StarrailMainStatValueGenerator()
    random_sub_stat_name = StarrailSubStatNameGenerator()
    random_sub_stat_value = StarrailSubStatValueGenerator()
    random_level = StarrailRelicLevelGenerator()
    random_equip = StarrailCharacterEquipGenerator()
    random_relic_count = StarrailRelicCountGenerator()
    random_number_generator = RandomNumberTextGenerator()

    weighted_generator = WeightedTextGenerator()
    weighted_generator.add_entry(0.1, random_relic_name)
    weighted_generator.add_entry(0.05, random_main_stat_name)
    weighted_generator.add_entry(0.15, random_main_stat_value)
    weighted_generator.add_entry(0.32, random_sub_stat_name)
    weighted_generator.add_entry(0.32, random_sub_stat_value)
    weighted_generator.add_entry(0.02, random_level)
    weighted_generator.add_entry(0.02, random_equip)
    weighted_generator.add_entry(0.1, random_relic_count)
    weighted_generator.add_entry(0.2, random_number_generator)

    return weighted_generator


def construct_all_lexicon():
    genshin = construct_genshin_lexicon()
    starrail = construct_starrail_lexicon()

    weighted_generator = WeightedTextGenerator()
    weighted_generator.add_entry(0.5, genshin)
    weighted_generator.add_entry(0.5, starrail)

    return weighted_generator


def get_lexicon(name: str):
    if name.lower() == "genshin":
        g = construct_genshin_lexicon()
    elif name.lower() == "starrail":
        g = construct_starrail_lexicon()
    else:
        g = construct_all_lexicon()

    return Lexicon(g)
