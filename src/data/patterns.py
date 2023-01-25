DEFAULT_NER_TAGS = [
    "CARDINAL", "DATE", "EVENT", "FAC", "GPE", 
    "LOC", "MONEY", "ORDINAL", "ORG", "PERCENT", 
    "PRODUCT", "QUANTITY", "TIME"
]


QUERY_PATTERNS = {
    "CrisisFACTS-General-q001": {"text": "Have airports closed", "keywords": "airport closed", "entities": ["GPE", "FAC", "LOC", "ORG"]},
    "CrisisFACTS-General-q002": {"text": "Have railways closed", "keywords": "rail closed", "entities": ["GPE", "FAC", "LOC", "ORG"]},
    "CrisisFACTS-General-q003": {"text": "Have water supplied been contaminated", "keywords": "water supply", "entities": ["GPE", "FAC", "LOC", "ORG"]},
    "CrisisFACTS-General-q004": {"text": "How many firefighters are active", "keywords": "firefighters on-duty", "entities": ["CARDINAL", "QUANTITY"]},
    "CrisisFACTS-General-q005": {"text": "How many people are affected", "keywords": "evacuated", "entities": ["CARDINAL", "QUANTITY"]},
    "CrisisFACTS-General-q006": {"text": "How many people are in shelters", "keywords": "shelters", "entities": ["CARDINAL", "QUANTITY"]},
    "CrisisFACTS-General-q007": {"text": "How many people are missing", "keywords": "missing", "entities": ["CARDINAL", "QUANTITY"]},
    "CrisisFACTS-General-q008": {"text": "How many people are trapped", "keywords": "trapped", "entities": ["CARDINAL", "QUANTITY"]},
    "CrisisFACTS-General-q009": {"text": "How many people have been injured", "keywords": "injury injured", "entities": ["CARDINAL", "QUANTITY"]},
    "CrisisFACTS-General-q010": {"text": "How many people have been killed", "keywords": "killed dead", "entities": ["CARDINAL", "QUANTITY"]},
    "CrisisFACTS-General-q011": {"text": "How much monetary damage has been caused", "keywords": "damage cost", "entities": ["CARDINAL", "MONEY", "QUANTITY", "PERCENT"]},
    "CrisisFACTS-General-q012": {"text": "How much rain-fall as occured", "keywords": "rain rainfall", "entities": ["CARDINAL", "QUANTITY"]},
    "CrisisFACTS-General-q013": {"text": "What are the wind speeds", "keywords": "wind speed mph", "entities": ["CARDINAL", "QUANTITY"]},
    "CrisisFACTS-General-q014": {"text": "What areas are being evacuated", "keywords": "evacuation", "entities": ["GPE", "FAC", "LOC"]},
    "CrisisFACTS-General-q015": {"text": "What areas are predicted to be impacted", "keywords": "forecast cone path", "entities": ["GPE", "FAC", "LOC"]},
    "CrisisFACTS-General-q016": {"text": "What areas are without power", "keywords": "electricity power", "entities": ["GPE", "FAC", "LOC"]},
    "CrisisFACTS-General-q017": {"text": "What at-risk groups are being impacted", "keywords": "Are there goods needing delivered", "entities": ["GPE", "FAC", "LOC", "PRODUCT"]},
    "CrisisFACTS-General-q018": {"text": "What barriers are hindering response efforts", "keywords": "smoke", "entities": ["GPE", "FAC", "LOC", "ORG"]},
    "CrisisFACTS-General-q019": {"text": "What bridges have been closed", "keywords": "bridges", "entities": ["GPE", "FAC", "LOC"]},
    "CrisisFACTS-General-q020": {"text": "What curfews are in place", "keywords": "curfew", "entities": ["GPE", "FAC", "LOC", "ORG"]},
    "CrisisFACTS-General-q021": {"text": "Are there goods needing delivered", "keywords": "food water transport", "entities": ["PRODUCT", "ORG"]},
    "CrisisFACTS-General-q022": {"text": "What events have been canceled", "keywords": "canceled", "entities": ["EVENT", "ORG"]},
    "CrisisFACTS-General-q023": {"text": "What goods are being requested", "keywords": "food water sandbags", "entities": ["PRODUCT", "ORG"]},
    "CrisisFACTS-General-q024": {"text": "What preparations are being made", "keywords": "various", "entities": ["GPE", "FAC", "LOC", "ORG"]},
    "CrisisFACTS-General-q025": {"text": "What regions have announced a state of emergency", "keywords": "state of emergency", "entities": ["GPE", "FAC", "LOC", "ORG"]},
    "CrisisFACTS-General-q026": {"text": "What roads are blocked \/ closed", "keywords": "tree block road closures", "entities": ["GPE", "FAC", "LOC"]},
    "CrisisFACTS-General-q027": {"text": "What roads have been re-opened", "keywords": "road re-opened", "entities": ["GPE", "FAC", "LOC"]},
    "CrisisFACTS-General-q028": {"text": "What services have been closed", "keywords": "closed", "entities": ["FAC", "ORG"]},
    "CrisisFACTS-General-q029": {"text": "What shelters are open", "keywords": "shelters", "entities": ["GPE", "FAC", "LOC", "ORG"]},
    "CrisisFACTS-General-q030": {"text": "What third-party support groups are active", "keywords": "SCEMD icrc", "entities": ["ORG"]},
    "CrisisFACTS-General-q031": {"text": "What traffic diversions are in effect", "keywords": "traffic", "entities": ["GPE", "FAC", "LOC", "ORG"]},
    "CrisisFACTS-General-q032": {"text": "What warnings are currently in effect", "keywords": "warning", "entities": ["GPE", "FAC", "LOC", "ORG"]},
    "CrisisFACTS-General-q033": {"text": "What watches are currently in effect", "keywords": "watch", "entities": ["GPE", "FAC", "LOC", "ORG"]},
    "CrisisFACTS-General-q034": {"text": "Where are emergency services deployed", "keywords": "service teams emergency services", "entities": ["GPE", "FAC", "LOC"]},
    "CrisisFACTS-General-q035": {"text": "Where are emergency services needed", "keywords": "ambulance medics", "entities": ["GPE", "FAC", "LOC"]},
    "CrisisFACTS-General-q036": {"text": "Where are evacuation centres", "keywords": "evacuation center shelters", "entities": ["GPE", "FAC", "LOC"]},
    "CrisisFACTS-General-q037": {"text": "Where are evacuations needed", "keywords": "evacuation evacs", "entities": ["GPE", "FAC", "LOC"]},
    "CrisisFACTS-General-q038": {"text": "Where are firefighters needed", "keywords": "fire fighters engines", "entities": ["GPE", "FAC", "LOC"]},
    "CrisisFACTS-General-q039": {"text": "Where are people needing rescued", "keywords": "rescue", "entities": ["GPE", "FAC", "LOC"]},
    "CrisisFACTS-General-q040": {"text": "Where are public officials located", "keywords": "PIO public information officer", "entities": ["GPE", "FAC", "LOC"]},
    "CrisisFACTS-General-q041": {"text": "Where are recovery efforts taking place", "keywords": "recovery effort", "entities": ["GPE", "FAC", "LOC"]},
    "CrisisFACTS-General-q042": {"text": "Where has building or infrastructure damage occurred", "keywords": "building damage", "entities": ["GPE", "FAC", "LOC"]},
    "CrisisFACTS-General-q043": {"text": "Where has flooding occured", "keywords": "flooding", "entities": ["GPE", "FAC", "LOC"]},
    "CrisisFACTS-General-q044": {"text": "Where are volunteers being requested", "keywords": "volunteer", "entities": ["GPE", "FAC", "LOC"]},
    "CrisisFACTS-General-q045": {"text": "What hazardous chemicals or materials are involved", "keywords": "fuel hazard waste infectious chemical", "entities": ["PRODUCT", "ORG"]},
    "CrisisFACTS-General-q046": {"text": "Where has road damage occured", "keywords": "road sinkhole", "entities": ["GPE", "FAC", "LOC"]},
    "CrisisFACTS-Wildfire-q001": {"text": "What area has the wildfire burned", "keywords": "acres size", "entities": ["GPE", "FAC", "LOC", "CARDINAL", "QUANTITY", "PERCENT"]},
    "CrisisFACTS-Wildfire-q002": {"text": "Where are wind speeds expected to be high", "keywords": "wind speed", "entities": ["GPE", "FAC", "LOC"]},
    "CrisisFACTS-Wildfire-q003": {"text": "Are helicopters available", "keywords": "helicopters", "entities": ["PRODUCT", "ORG"]},
    "CrisisFACTS-Wildfire-q004": {"text": "Where have homes been damaged or destroyed", "keywords": "homes destroyed damaged", "entities": ["GPE", "FAC", "LOC"]},
    "CrisisFACTS-Wildfire-q005": {"text": "How quickly is the fire spreading", "keywords": "acres per hour", "entities": ["CARDINAL", "QUANTITY"]},
    "CrisisFACTS-Wildfire-q006": {"text": "What is the fire containment level", "keywords": "containment", "entities": ["CARDINAL", "QUANTITY", "PERCENT"]},
    "CrisisFACTS-Hurricane-q001": {"text": "What is the hurricane category", "keywords": "category", "entities": ["EVENT", "CARDINAL", "QUANTITY"]},
    "CrisisFACTS-Hurricane-q002": {"text": "What is the hurricane pressure", "keywords": "pressure", "entities": ["EVENT", "CARDINAL", "QUANTITY"]},
    "CrisisFACTS-Hurricane-q003": {"text": "What direction is the hurricane moving", "keywords": "forecast north south east west", "entities": ["EVENT", "GPE", "FAC", "LOC"]},
    "CrisisFACTS-Hurricane-q004": {"text": "Where has the hurricane made landfall", "keywords": "landfall", "entities": ["EVENT", "GPE", "FAC", "LOC"]},
    "CrisisFACTS-Hurricane-q06": {"text": "How fast is the hurricane travelling", "keywords": "speed", "entities": ["EVENT", "CARDINAL", "QUANTITY"]},
    "CrisisFACTS-Flood-q001": {"text": "What flood warnings are active", "keywords": "flood warning", "entities": ["GPE", "FAC", "LOC", "ORG"]},
    "CrisisFACTS-Flood-q002": {"text": "What rivers have overflowed", "keywords": "river", "entities": ["GPE", "FAC", "LOC"]}
}