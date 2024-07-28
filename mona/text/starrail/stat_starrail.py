import json
import random
from mona.text.text_generator import TextGenerator


stat_info = {
    "FIGHT_PROP_HP": {
        "percent": False,
        "chs": "生命值",
    },
    "FIGHT_PROP_HP_PERCENTAGE": {
        "percent": True,
        "chs": "生命值",
    },
    "FIGHT_PROP_ATK": {
        "percent": False,
        "chs": "攻击力",
    },
    "FIGHT_PROP_ATK_PERCENTAGE": {
        "percent": True,
        "chs": "攻击力",
    },
    "FIGHT_PROP_DEF_PERCENTAGE": {
        "percent": True,
        "chs": "防御力",
    },
    "FIGHT_PROP_SPD": {
        "percent": False,
        "chs": "速度",
    },
    "FIGHT_PROP_CRIT_RATE": {
        "percent": True,
        "chs": "暴击率",
    },
    "FIGHT_PROP_CRIT_DMG": {
        "percent": True,
        "chs": "暴击伤害",
    },
    "FIGHT_PROP_BREAK_EFFECT": {
        "percent": True,
        "chs": "击破特攻",
    },
    "FIGHT_PROP_OUTGOING_HEALING_BOOST": {
        "percent": True,
        "chs": "治疗量加成",
    },
    "FIGHT_PROP_ENERGY_REGENERATION_RATE": {
        "percent": True,
        "chs": "能量恢复效率",
    },
    "FIGHT_PROP_EFFECT_HIT_RATE": {
        "percent": True,
        "chs": "效果命中",
    },
    "FIGHT_PROP_PHYSICAL_DMG_BOOST": {
        "percent": True,
        "chs": "物理属性伤害提高",
    },
    "FIGHT_PROP_FIRE_DMG_BOOST": {
        "percent": True,
        "chs": "火属性伤害提高",
    },
    "FIGHT_PROP_ICE_DMG_BOOST": {
        "percent": True,
        "chs": "冰属性伤害提高",
    },
    "FIGHT_PROP_LIGHTNING_DMG_BOOST": {
        "percent": True,
        "chs": "雷属性伤害提高",
    },
    "FIGHT_PROP_WIND_DMG_BOOST": {
        "percent": True,
        "chs": "风属性伤害提高",
    },
    "FIGHT_PROP_QUANTUM_DMG_BOOST": {
        "percent": True,
        "chs": "量子属性伤害提高",
    },
    "FIGHT_PROP_IMAGINARY_DMG_BOOST": {
        "percent": True,
        "chs": "虚数属性伤害提高",
    },
    "FIGHT_PROP_DEF": {
        "percent": False,
        "chs": "防御力",
    },
    "FIGHT_PROP_EEFFECT_RES": {
        "percent": True,
        "chs": "效果抵抗",
    }
}

stat_name_set = set()
for item in stat_info:
    stat_name_set.add(stat_info[item]["chs"])
stat_name = list(stat_name_set)

main_stat_names = {
    "head": ["FIGHT_PROP_HP"],
    "hands": ["FIGHT_PROP_ATK"],
    "body": ["FIGHT_PROP_HP_PERCENTAGE", "FIGHT_PROP_ATK_PERCENTAGE", "FIGHT_PROP_DEF_PERCENTAGE",  "FIGHT_PROP_CRIT_RATE", "FIGHT_PROP_CRIT_DMG", "FIGHT_PROP_OUTGOING_HEALING_BOOST", "FIGHT_PROP_EFFECT_HIT_RATE"],
    "feet": ["FIGHT_PROP_HP_PERCENTAGE", "FIGHT_PROP_ATK_PERCENTAGE",
             "FIGHT_PROP_DEF_PERCENTAGE", "FIGHT_PROP_SPD"],
    "planar_sphere": ["FIGHT_PROP_HP_PERCENTAGE", "FIGHT_PROP_ATK_PERCENTAGE",
                      "FIGHT_PROP_DEF_PERCENTAGE", "FIGHT_PROP_PHYSICAL_DMG_BOOST",
                      "FIGHT_PROP_FIRE_DMG_BOOST", "FIGHT_PROP_ICE_DMG_BOOST",
                      "FIGHT_PROP_WIND_DMG_BOOST", "FIGHT_PROP_LIGHTNING_DMG_BOOST",
                      "FIGHT_PROP_QUANTUM_DMG_BOOST", "FIGHT_PROP_IMAGINARY_DMG_BOOST"],
    "link_rope": ["FIGHT_PROP_HP_PERCENTAGE", "FIGHT_PROP_ATK_PERCENTAGE",
                  "FIGHT_PROP_DEF_PERCENTAGE", "FIGHT_PROP_BREAK_EFFECT",
                  "FIGHT_PROP_ENERGY_REGENERATION_RATE"]
}

sub_stat_keys = [
    "FIGHT_PROP_HP", 
    "FIGHT_PROP_HP_PERCENTAGE", 
    "FIGHT_PROP_ATK", 
    "FIGHT_PROP_ATK_PERCENTAGE", 
    "FIGHT_PROP_DEF_PERCENTAGE", 
    "FIGHT_PROP_SPD", 
    "FIGHT_PROP_CRIT_RATE", 
    "FIGHT_PROP_CRIT_DMG", 
    "FIGHT_PROP_BREAK_EFFECT", 
    "FIGHT_PROP_EFFECT_HIT_RATE", 
    "FIGHT_PROP_DEF", 
    "FIGHT_PROP_EEFFECT_RES"
]

main_stat_keys = [
    "FIGHT_PROP_HP", 
    "FIGHT_PROP_HP_PERCENTAGE", 
    "FIGHT_PROP_ATK", 
    "FIGHT_PROP_ATK_PERCENTAGE", 
    "FIGHT_PROP_DEF_PERCENTAGE", 
    "FIGHT_PROP_SPD", 
    "FIGHT_PROP_CRIT_RATE", 
    "FIGHT_PROP_CRIT_DMG", 
    "FIGHT_PROP_BREAK_EFFECT", 
    "FIGHT_PROP_OUTGOING_HEALING_BOOST", 
    "FIGHT_PROP_ENERGY_REGENERATION_RATE", 
    "FIGHT_PROP_EFFECT_HIT_RATE", 
    "FIGHT_PROP_PHYSICAL_DMG_BOOST", 
    "FIGHT_PROP_FIRE_DMG_BOOST", 
    "FIGHT_PROP_ICE_DMG_BOOST", 
    "FIGHT_PROP_LIGHTNING_DMG_BOOST", 
    "FIGHT_PROP_WIND_DMG_BOOST", 
    "FIGHT_PROP_QUANTUM_DMG_BOOST", 
    "FIGHT_PROP_IMAGINARY_DMG_BOOST"
]

# 副词条大致范围，超过理论值也没问题
sub_stat_range = {
    "FIGHT_PROP_HP": (10, 300),
    "FIGHT_PROP_HP_PERCENTAGE": (0.001, 0.3),
    "FIGHT_PROP_ATK": (1, 150),    
    "FIGHT_PROP_ATK_PERCENTAGE": (0.001, 0.3),
    "FIGHT_PROP_DEF_PERCENTAGE": (0.001, 0.4),
    "FIGHT_PROP_SPD": (1, 20),
    "FIGHT_PROP_CRIT_RATE": (0.001, 0.25),
    "FIGHT_PROP_CRIT_DMG": (0.001, 0.5),    
    "FIGHT_PROP_BREAK_EFFECT": (0.001, 0.4),
    "FIGHT_PROP_EFFECT_HIT_RATE": (0.001, 0.3),
    "FIGHT_PROP_DEF": (1, 150),
    "FIGHT_PROP_EEFFECT_RES": (0.001, 0.3)
}

with open("./assets/RelicLevelExcelConfigData.json") as f:
    string = f.read()
    main_stat_data = json.loads(string)

main_stat_map = {}
for item in main_stat_data:
    star = item["Rank"]
    level = item["Level"] - 1
    data = item["AddProps"]

    if star not in main_stat_map:
        main_stat_map[star] = {}
    main_stat_map[star][level] = {}
    for i in data:
        key = i["PropType"]
        value = i["Value"]
        main_stat_map[star][level][key] = value


def format_value(stat_name, value):
    if stat_info[stat_name]["percent"]:
        return str(round(value * 100, 1)) + "%"
    else:
        temp = str(int(value))
        if len(temp) >= 4:
            temp = temp[0] + "," + temp[1:]
        return temp


class StarrailMainStatNameGenerator(TextGenerator):
    def __init__(self):
        super(StarrailMainStatNameGenerator, self).__init__("Starrail Main Stat Name")

    def generate_text(self):
        position = random.choice(list(main_stat_names.keys()))
        entry = random.choice(main_stat_names[position])
        return stat_info[entry]["chs"]

    def get_lexicon(self):
        ret = set()
        for k in stat_info:
            for c in stat_info[k]["chs"]:
                ret.add(c)
        return ret


class StarrailMainStatValueGenerator(TextGenerator):
    def __init__(self):
        super(StarrailMainStatValueGenerator, self).__init__("Starrail Main Stat Value")

    def generate_text(self):
        position = random.choice(list(main_stat_names.keys()))
        key = random.choice(main_stat_names[position])

        star = random.choices(population=[1, 2, 3, 4, 5], weights=[0.05, 0.05, 0.2, 0.2, 0.5], k=1)[0]
        if star == 5:
            level = random.randint(0, 15)
        elif star == 4:
            level = random.randint(0, 12)
        elif star == 3:
            level = random.randint(0, 9)
        else:
            level = 0

        value = main_stat_map[star][level][key]

        return format_value(key, value)

    def get_lexicon(self):
        ret = set()
        for c in " '0123456789.+%,/":
            ret.add(c)
        return ret


class StarrailSubStatNameGenerator(TextGenerator):
    def __init__(self):
        super(StarrailSubStatNameGenerator, self).__init__("Starrail Sub Stat Name")

    def generate_text(self):
        key = random.choice(sub_stat_keys)
        chs = stat_info[key]["chs"]
        return chs

    def get_lexicon(self):
        ret = set()
        for k in stat_info:
            for c in stat_info[k]["chs"]:
                ret.add(c)
        return ret


class StarrailSubStatValueGenerator(TextGenerator):
    def __init__(self):
        super(StarrailSubStatValueGenerator, self).__init__("Starrail Sub Stat Value")

    def generate_text(self):
        key = random.choice(sub_stat_keys)
        # 改成生成连续数值。因为原版数据有误差，生成不出777、299等数值
        value = random.uniform(sub_stat_range[key][0], sub_stat_range[key][1])
        value_str = format_value(key, value)
        return value_str

    def get_lexicon(self):
        ret = set()
        for c in " '0123456789.+%,/":
            ret.add(c)
        return ret
