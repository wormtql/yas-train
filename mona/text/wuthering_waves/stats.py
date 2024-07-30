import random

from mona.text.text_generator import TextGenerator


main_stat_names = [
    "暴击率",
    "暴击伤害",
    "冷凝伤害加成",
    "热熔伤害加成",
    "导电伤害加成",
    "气动伤害加成",
    "衍射伤害加成",
    "湮灭伤害加成",
    "共鸣效率",
    "攻击",
    "治疗效果加成",
    "生命",
    "防御",
]

# todo
main_stat_values = [
    # atk% yellow
    "", "", "", "", "6.4%"
]


class WWMainStatNameGenerator(TextGenerator):
    def __init__(self):
        super(WWMainStatNameGenerator, self).__init__("Wuthering Waves Main Stat Name")

    def generate_text(self):
        return random.choice(main_stat_names)

    def get_lexicon(self):
        ret = set()
        for name in main_stat_names:
            for c in name:
                ret.add(c)
        return ret


class WWMainStatValueGenerator(TextGenerator):
    def __init__(self):
        super(WWMainStatValueGenerator, self).__init__("Wuthering Waves Main Stat Value")

    def generate_text(self):
        # todo
        value = random.randint(0, 9999)
        if random.random() < 0.5:
            return f"{value}"
        else:
            x = value // 100
            if x == 0:
                return f"{value}"
            else:
                y = (value % 100) // 10
                return f"{x}.{y}%"

    def get_lexicon(self):
        ret = set()
        for c in "0123456789%.":
            ret.add(c)
        return ret


sub_stat_names = [
    "暴击率",
    "暴击伤害",
    "攻击",
    "生命",
    "防御",
    "普攻伤害加成",
    "重击伤害加成",
    "共鸣技能伤害加成",
    "共鸣解放伤害加成",
    "共鸣效率"
]

sub_stat_values = [
    # critical rate
    "10.5%",
    "9.9%",
    "9.3%",
    "8.7%",
    "8.4%",
    "8.1%",
    "7.5%",
    "6.9%",
    "6.3%",
    # critical damage
    "21%",
    "19.8%",
    "18.6%",
    "17.4%",
    "16.2%",
    "15.0%",
    "13.8%",
    "12.6%",
    # %atk ...
    "11.6%",
    "10.9%",
    "10.1%",
    "9.4%",
    "8.6%",
    "7.9%",
    "7.1%",
    "6.4%",
    "6.0%",
    # %def
    "14.7%",
    "13.8%",
    "12.8%",
    "11.8%",
    "10.9%",
    "10.0%",
    "9.0%",
    "8.1%",
    # recharge
    "12.4%",
    "11.6%",
    "10.8%",
    "10.0%",
    "9.2%",
    "8.4%",
    "7.6%",
    "6.8%",
    # HP
    "580",
    "540",
    "510",
    "470",
    "430",
    "390",
    "360",
    "320",
    # ATK & DEF
    "60",
    "50",
    "40"
]


class WWSubStatNameGenerator(TextGenerator):
    def __init__(self):
        super(WWSubStatNameGenerator, self).__init__("Wuthering Waves Sub Stat Name")

    def generate_text(self):
        return random.choice(sub_stat_names)

    def get_lexicon(self):
        ret = set()
        for name in sub_stat_names:
            for c in name:
                ret.add(c)
        return ret


class WWSubStatValueGenerator(TextGenerator):
    def __init__(self):
        super(WWSubStatValueGenerator, self).__init__("Wuthering Waves Sub Stat Value")

    def generate_text(self):
        return random.choice(sub_stat_values)

    def get_lexicon(self):
        ret = set()
        for item in sub_stat_values:
            for c in item:
                ret.add(c)
        return ret
