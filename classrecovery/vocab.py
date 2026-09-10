"""Built-in vocabulary of concrete object nouns for prompt search and CLIP naming.

The list is deliberately broad rather than deep: it is a starting point for
locating the *neighbourhood* of a class, after which a domain-specific
vocabulary (``--vocab my_words.txt``, one term per line) usually does better.
"""

from __future__ import annotations

from typing import Iterable, List

COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]

ANIMALS = [
    "lion", "tiger", "leopard", "cheetah", "wolf", "fox", "deer", "moose", "elk", "antelope", "gazelle",
    "buffalo", "bison", "rhinoceros", "hippopotamus", "camel", "llama", "goat", "pig", "donkey", "monkey",
    "gorilla", "chimpanzee", "kangaroo", "koala", "panda", "raccoon", "squirrel", "rabbit", "hamster", "mouse",
    "rat", "hedgehog", "bat", "otter", "seal", "walrus", "dolphin", "whale", "shark", "fish", "octopus",
    "crab", "lobster", "jellyfish", "turtle", "tortoise", "snake", "lizard", "crocodile", "alligator", "frog",
    "chicken", "rooster", "duck", "goose", "swan", "turkey", "peacock", "owl", "eagle", "hawk", "parrot",
    "penguin", "flamingo", "pigeon", "butterfly", "bee", "ant", "spider", "beetle", "dragonfly", "mosquito",
]

VEHICLES = [
    "sedan", "suv", "pickup truck", "van", "minivan", "jeep", "convertible", "limousine", "taxi", "police car",
    "ambulance", "fire truck", "garbage truck", "tow truck", "dump truck", "cement mixer", "semi truck",
    "tanker truck", "tractor", "combine harvester", "bulldozer", "excavator", "forklift", "crane", "backhoe",
    "road roller", "golf cart", "scooter", "moped", "tricycle", "wheelchair", "stroller", "shopping cart",
    "tank", "armored vehicle", "humvee", "military truck", "artillery", "missile launcher", "helicopter",
    "fighter jet", "drone", "quadcopter", "airliner", "cargo plane", "glider", "hot air balloon", "blimp",
    "rocket", "satellite", "space shuttle", "sailboat", "yacht", "speedboat", "canoe", "kayak", "jet ski",
    "cargo ship", "container ship", "cruise ship", "aircraft carrier", "warship", "submarine", "ferry",
    "fishing boat", "tugboat", "barge", "locomotive", "tram", "subway car", "monorail", "cable car",
]

AERIAL = [
    "building", "house", "skyscraper", "warehouse", "factory", "stadium", "parking lot", "swimming pool",
    "tennis court", "basketball court", "baseball diamond", "soccer field", "running track", "roundabout",
    "bridge", "overpass", "highway", "railway", "airport runway", "helipad", "harbor", "pier", "dam",
    "storage tank", "silo", "wind turbine", "solar panel", "power line", "transmission tower", "chimney",
    "shipping container", "greenhouse", "farmland", "orchard", "vineyard", "forest", "lake", "river",
    "construction site", "oil rig", "radar dome", "antenna", "water tower", "lighthouse", "campsite", "tent",
]

HOUSEHOLD = [
    "table", "desk", "stool", "sofa", "armchair", "bookshelf", "cabinet", "drawer", "wardrobe", "mirror",
    "lamp", "chandelier", "ceiling fan", "curtain", "rug", "pillow", "blanket", "towel", "mattress", "crib",
    "bathtub", "shower", "faucet", "washing machine", "dryer", "dishwasher", "kettle", "coffee maker",
    "blender", "mixer", "pan", "pot", "plate", "mug", "glass", "jar", "can", "box", "basket", "bucket",
    "broom", "mop", "vacuum cleaner", "iron", "sewing machine", "candle", "picture frame", "painting",
    "television", "monitor", "speaker", "headphones", "camera", "tablet", "smartphone", "game controller",
    "printer", "router", "light switch", "power outlet", "door", "window", "stairs", "fireplace", "radiator",
]

TOOLS_AND_INDUSTRY = [
    "hammer", "screwdriver", "wrench", "pliers", "saw", "drill", "chainsaw", "axe", "shovel", "rake",
    "ladder", "toolbox", "tape measure", "level", "nail", "screw", "bolt", "nut", "gear", "pipe", "valve",
    "gauge", "circuit board", "microchip", "battery", "cable", "wire", "generator", "engine", "motor",
    "pump", "compressor", "welding torch", "fire extinguisher", "hard hat", "safety vest", "gas mask",
    "traffic cone", "barrier", "pallet", "barrel", "crate", "conveyor belt", "robot arm", "3d printer",
]

PEOPLE_AND_CLOTHING = [
    "man", "woman", "child", "baby", "soldier", "police officer", "firefighter", "doctor", "nurse", "chef",
    "construction worker", "athlete", "cyclist", "pedestrian", "crowd", "face", "hand", "hat", "helmet", "cap",
    "glasses", "sunglasses", "mask", "scarf", "gloves", "jacket", "coat", "shirt", "t-shirt", "dress", "skirt",
    "pants", "jeans", "shorts", "shoes", "boots", "sneakers", "sandals", "socks", "belt", "watch", "ring",
    "necklace", "earrings", "bracelet", "wallet", "purse", "briefcase", "uniform", "vest", "badge",
]

FOOD = [
    "bread", "croissant", "bagel", "muffin", "cookie", "pie", "ice cream", "chocolate", "candy", "cheese",
    "egg", "bacon", "sausage", "steak", "chicken wing", "hamburger", "french fries", "taco", "burrito", "sushi",
    "noodles", "rice", "salad", "soup", "strawberry", "grape", "watermelon", "pineapple", "mango", "lemon",
    "peach", "pear", "cherry", "tomato", "potato", "onion", "garlic", "pepper", "cucumber", "lettuce",
    "corn", "mushroom", "pumpkin", "avocado", "coffee", "tea", "beer", "wine", "juice", "milk", "water bottle",
]

WEAPONS_AND_SECURITY = [
    "gun", "pistol", "rifle", "shotgun", "knife", "sword", "bow", "grenade", "bullet", "ammunition",
    "handcuffs", "security camera", "surveillance camera", "metal detector", "x-ray scanner", "baton",
    "shield", "body armor", "binoculars", "walkie talkie", "flashlight", "license plate", "road sign",
]

MEDICAL_AND_MISC = [
    "syringe", "pill", "bandage", "stethoscope", "thermometer", "microscope", "test tube", "wheelchair",
    "crutch", "hospital bed", "flower", "tree", "bush", "cactus", "leaf", "rock", "mountain", "cloud", "sun",
    "moon", "star", "fire", "smoke", "flag", "sign", "poster", "billboard", "logo", "text", "barcode",
    "qr code", "coin", "banknote", "credit card", "key", "lock", "trash can", "mailbox", "fence", "gate",
    "playground", "swing", "slide", "trampoline", "kite", "balloon", "gift", "toy", "doll", "lego", "puzzle",
    "guitar", "piano", "drum", "violin", "trumpet", "microphone", "ball", "football", "basketball",
    "tennis ball", "golf ball", "hockey stick", "bicycle helmet", "skate", "surfboard", "fishing rod",
]


def default_vocab() -> List[str]:
    seen, out = set(), []
    for group in (COCO80, ANIMALS, VEHICLES, AERIAL, HOUSEHOLD, TOOLS_AND_INDUSTRY,
                  PEOPLE_AND_CLOTHING, FOOD, WEAPONS_AND_SECURITY, MEDICAL_AND_MISC):
        for w in group:
            if w not in seen:
                seen.add(w)
                out.append(w)
    return out


def load_vocab(path: str | None = None, extra: Iterable[str] = ()) -> List[str]:
    words = default_vocab() if path is None else [
        ln.strip() for ln in open(path, encoding="utf-8") if ln.strip() and not ln.startswith("#")
    ]
    for w in extra:
        if w and w not in words:
            words.append(w)
    return words


def make_prompt(word: str, template: str = "a photo of a {}") -> str:
    article_fix = template.replace("a {}", "an {}") if word[:1].lower() in "aeiou" and "a {}" in template else template
    return article_fix.format(word)
