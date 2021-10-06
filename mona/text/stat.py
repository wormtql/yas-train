import json
import random


stat_info = {
    "FIGHT_PROP_CRITICAL_HURT": {
        "percent": True,
        "chs": "暴击伤害",
    },
    "FIGHT_PROP_CRITICAL": {
        "percent": True,
        "chs": "暴击率",
    },
    "FIGHT_PROP_HP": {
        "percent": False,
        "chs": "生命值",
    },
    "FIGHT_PROP_HP_PERCENT": {
        "percent": True,
        "chs": "生命值",
    },
    "FIGHT_PROP_ATTACK": {
        "percent": False,
        "chs": "攻击力",
    },
    "FIGHT_PROP_ATTACK_PERCENT": {
        "percent": True,
        "chs": "攻击力",
    },
    "FIGHT_PROP_DEFENSE": {
        "percent": False,
        "chs": "防御力",
    },
    "FIGHT_PROP_DEFENSE_PERCENT": {
        "percent": True,
        "chs": "防御力",
    },
    "FIGHT_PROP_CHARGE_EFFICIENCY": {
        "percent": True,
        "chs": "元素充能效率",
    },
    "FIGHT_PROP_ELEMENT_MASTERY": {
        "percent": False,
        "chs": "元素精通",
    },
    "FIGHT_PROP_HEAL_ADD": {
        "percent": True,
        "chs": "治疗加成",
    },
    "FIGHT_PROP_FIRE_ADD_HURT": {
        "percent": True,
        "chs": "火元素伤害加成",
    },
    "FIGHT_PROP_ELEC_ADD_HURT": {
        "percent": True,
        "chs": "雷元素伤害加成",
    },
    "FIGHT_PROP_WATER_ADD_HURT": {
        "percent": True,
        "chs": "水元素伤害加成",
    },
    "FIGHT_PROP_WIND_ADD_HURT": {
        "percent": True,
        "chs": "风元素伤害加成",
    },
    "FIGHT_PROP_ROCK_ADD_HURT": {
        "percent": True,
        "chs": "岩元素伤害加成",
    },
    # "FIGHT_PROP_GRASS_ADD_HURT": {},
    "FIGHT_PROP_ICE_ADD_HURT": {
        "percent": True,
        "chs": "冰元素伤害加成",
    },
    "FIGHT_PROP_PHYSICAL_ADD_HURT": {
        "percent": True,
        "chs": "物理伤害加成",
    }
}

stat_name_set = set()
for item in stat_info:
    stat_name_set.add(stat_info[item]["chs"])
stat_name = list(stat_name_set)

main_stat_names = {
    "flower": ["FIGHT_PROP_HP"],
    "feather": ["FIGHT_PROP_ATTACK"],
    "sand": ["FIGHT_PROP_ELEMENT_MASTERY", "FIGHT_PROP_CHARGE_EFFICIENCY",
             "FIGHT_PROP_DEFENSE_PERCENT", "FIGHT_PROP_ATTACK_PERCENT", "FIGHT_PROP_HP_PERCENT"],
    "cup": ["FIGHT_PROP_PHYSICAL_ADD_HURT", "FIGHT_PROP_ICE_ADD_HURT", "FIGHT_PROP_ROCK_ADD_HURT",
            "FIGHT_PROP_WIND_ADD_HURT", "FIGHT_PROP_WATER_ADD_HURT", "FIGHT_PROP_ELEC_ADD_HURT",
            "FIGHT_PROP_FIRE_ADD_HURT", "FIGHT_PROP_ELEMENT_MASTERY", "FIGHT_PROP_DEFENSE_PERCENT",
            "FIGHT_PROP_ATTACK_PERCENT", "FIGHT_PROP_HP_PERCENT"],
    "head": ["FIGHT_PROP_HEAL_ADD", "FIGHT_PROP_ELEMENT_MASTERY", "FIGHT_PROP_DEFENSE_PERCENT",
             "FIGHT_PROP_ATTACK_PERCENT", "FIGHT_PROP_HP_PERCENT", "FIGHT_PROP_CRITICAL",
             "FIGHT_PROP_CRITICAL_HURT"]
}

sub_stat_keys = [
    "FIGHT_PROP_CRITICAL_HURT", "FIGHT_PROP_CRITICAL", "FIGHT_PROP_HP", "FIGHT_PROP_HP_PERCENT",
    "FIGHT_PROP_ATTACK", "FIGHT_PROP_ATTACK_PERCENT", "FIGHT_PROP_DEFENSE_PERCENT", "FIGHT_PROP_DEFENSE",
    "FIGHT_PROP_CHARGE_EFFICIENCY", "FIGHT_PROP_ELEMENT_MASTERY",
]

main_stat_keys = [
    "FIGHT_PROP_CRITICAL_HURT", "FIGHT_PROP_CRITICAL", "FIGHT_PROP_HP", "FIGHT_PROP_HP_PERCENT",
    "FIGHT_PROP_ATTACK", "FIGHT_PROP_ATTACK_PERCENT", "FIGHT_PROP_DEFENSE_PERCENT", "FIGHT_PROP_DEFENSE",
    "FIGHT_PROP_CHARGE_EFFICIENCY", "FIGHT_PROP_ELEMENT_MASTERY", "FIGHT_PROP_HEAL_ADD",
    "FIGHT_PROP_PHYSICAL_ADD_HURT", "FIGHT_PROP_ICE_ADD_HURT", "FIGHT_PROP_ROCK_ADD_HURT",
    "FIGHT_PROP_WIND_ADD_HURT", "FIGHT_PROP_WATER_ADD_HURT", "FIGHT_PROP_ELEC_ADD_HURT",
    "FIGHT_PROP_FIRE_ADD_HURT"
]

with open("./assets/ReliquaryLevelExcelConfigData.json") as f:
    string = f.read()
    main_stat_data = json.loads(string)
with open("./assets/ReliquaryAffixExcelConfigData.json") as f:
    string = f.read()
    sub_stat_data = json.loads(string)

sub_stat_map = {}
for item in sub_stat_data:
    star = item["DepotId"] // 100
    key = item["PropType"]
    value = item["PropValue"]

    if key not in sub_stat_map:
        sub_stat_map[key] = {}

    if star not in sub_stat_map[key]:
        sub_stat_map[key][star] = []

    sub_stat_map[key][star].append(value)


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


def random_main_stat_name():
    position = random.choice(list(main_stat_names.keys()))
    entry = random.choice(main_stat_names[position])
    return stat_info[entry]["chs"]


def random_main_stat_value():
    position = random.choice(list(main_stat_names.keys()))
    key = random.choice(main_stat_names[position])

    star = random.choices(population=[1, 2, 3, 4, 5], weights=[0.05, 0.05, 0.2, 0.2, 0.5], k=1)[0]
    if star == 5:
        level = random.randint(0, 20)
    elif star == 4:
        level = random.randint(0, 16)
    elif star == 3:
        level = random.randint(0, 16)
    else:
        level = 0

    value = main_stat_map[star][level][key]

    return format_value(key, value)


def random_sub_stat():
    key = random.choice(sub_stat_keys)
    star = random.choices(population=[1, 2, 3, 4, 5], weights=[0.05, 0.05, 0.2, 0.2, 0.5], k=1)[0]
    r = sub_stat_map[key][star]

    if star == 5:
        times = random.randint(1, 6)
    else:
        times = random.randint(1, 5)

    value = 0
    for _ in range(times):
        value += random.choice(r)

    value_str = format_value(key, value)
    chs = stat_info[key]["chs"]

    return chs + "+" + value_str
