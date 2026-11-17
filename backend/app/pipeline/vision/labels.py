"""Controlled vocabularies and prompts for zero-shot annotation. Editing these changes the
zero-shot stage fingerprint, so affected images are re-annotated (embeddings are reused)."""

# name -> [(label, prompt)]
LABEL_SETS = {
    "categories": [
        ("landscape", "a landscape photo of natural scenery"),
        ("architecture", "a photo of a building or architecture"),
        ("cityscape", "a photo of a city street or skyline"),
        ("interior", "a photo of an indoor room or interior"),
        ("wildlife", "a photo of a wild animal"),
        ("pet", "a photo of a pet dog or cat"),
        ("vehicle", "a photo of a car, vehicle or other transport"),
        ("food", "a photo of food or a drink"),
        ("portrait", "a portrait photo of a person"),
        ("people", "a photo of a group of people"),
        ("product", "a product photo of an object on a plain background"),
        ("plant", "a close-up photo of a plant or flower"),
        ("sports", "a photo of people playing sports"),
        ("illustration", "a drawn or painted illustration"),
        ("abstract", "an abstract image of shapes, patterns or textures"),
        ("document", "a screenshot or a scanned document with text"),
        ("graphic", "a graphic design, logo or poster"),
        ("aerial", "an aerial photo taken from above"),
        ("night_sky", "a photo of the night sky with stars"),
        ("underwater", "an underwater photo"),
    ],
    "style": [
        ("photorealistic", "a realistic photograph"),
        ("cinematic", "a cinematic film still"),
        ("black_and_white_photo", "a black and white photograph"),
        ("digital_illustration", "a digital illustration"),
        ("anime", "an anime or manga style drawing"),
        ("3d_render", "a 3D computer rendered image"),
        ("oil_painting", "an oil painting"),
        ("watercolor", "a watercolor painting"),
        ("sketch", "a pencil sketch or line drawing"),
        ("graphic_design", "a flat graphic design or vector art"),
        ("pixel_art", "pixel art"),
    ],
    "lighting": [
        ("daylight", "a photo taken in bright daylight"),
        ("golden_hour", "a photo taken at golden hour with warm low sunlight"),
        ("backlit", "a backlit photo with a silhouette"),
        ("studio_lighting", "a photo with even studio lighting"),
        ("low_light", "a dark photo taken in low light at night"),
        ("overcast", "a photo taken under soft overcast light"),
        ("indoor_artificial", "a photo lit by indoor artificial light"),
        ("neon", "a photo lit by colorful neon lights"),
        ("flat_no_lighting", "a flat image with no visible lighting"),
    ],
    "scene": [
        ("indoor", "an indoor scene"),
        ("outdoor_urban", "an outdoor urban scene"),
        ("outdoor_natural", "an outdoor scene in nature"),
        ("studio_backdrop", "an object in front of a plain studio backdrop"),
        ("no_setting", "a flat image with no real-world setting"),
    ],
    "viewpoint": [
        ("eye_level", "a photo taken at eye level"),
        ("low_angle", "a photo taken from a low angle looking up"),
        ("high_angle", "a photo taken from a high angle looking down"),
        ("top_down", "a top-down overhead view"),
        ("close_up", "an extreme close-up macro shot"),
        ("wide_shot", "a wide shot showing a large area"),
    ],
    "depth_of_field": [
        ("shallow", "a photo with a blurred background and shallow depth of field"),
        ("deep", "a photo where everything is in sharp focus"),
    ],
}

MULTI_LABEL_MIN = {"categories": 0.20, "style": 0.30, "lighting": 0.30, "scene": 0.35, "viewpoint": 0.35, "depth_of_field": 0.6}
MAX_LABELS = {"categories": 3, "style": 2, "lighting": 2, "scene": 1, "viewpoint": 1, "depth_of_field": 1}

AESTHETIC_PAIRS = [
    ("a beautiful, well composed photograph", "an ugly, poorly composed photograph"),
    ("a high quality professional image", "a low quality amateur image"),
    ("an image with pleasing colors and lighting", "an image with unpleasant colors and bad lighting"),
    ("a visually striking image", "a dull, boring image"),
]

WATERMARK_PROMPTS = [
    "an image with a watermark text overlay",
    "a stock photo with a visible watermark logo",
    "a clean image without any watermark",
]

DETECTION_LABELS = ["person", "face", "dog", "cat", "bird", "horse", "car", "bicycle", "motorcycle", "bus", "truck",
                    "boat", "airplane", "tree", "flower", "building", "mountain", "chair", "table", "bottle", "cup",
                    "plate", "book", "laptop", "phone", "sign", "logo"]

# Words a caption may use for each label (used by caption grounding checks and tags).
CATEGORY_TERMS = {
    "landscape": ["landscape", "scenery", "mountain", "mountains", "valley", "hills", "field", "coast", "beach", "desert", "forest", "lake", "river"],
    "architecture": ["building", "buildings", "architecture", "house", "tower", "bridge", "church", "facade", "skyscraper"],
    "cityscape": ["city", "street", "skyline", "downtown", "urban", "road", "traffic"],
    "interior": ["room", "interior", "kitchen", "bedroom", "office", "living room", "hallway", "indoors"],
    "wildlife": ["animal", "wildlife", "bird", "deer", "lion", "elephant", "bear", "fox", "insect"],
    "pet": ["dog", "cat", "puppy", "kitten", "pet"],
    "vehicle": ["car", "vehicle", "truck", "bus", "motorcycle", "bicycle", "train", "boat", "airplane", "plane"],
    "food": ["food", "meal", "dish", "plate", "drink", "coffee", "fruit", "dessert", "pizza", "salad", "cake"],
    "portrait": ["person", "man", "woman", "portrait", "face", "girl", "boy", "child"],
    "people": ["people", "group", "crowd", "men", "women", "children", "team", "family"],
    "product": ["product", "bottle", "box", "package", "device", "watch", "shoe", "gadget"],
    "plant": ["plant", "flower", "flowers", "leaf", "leaves", "tree", "blossom"],
    "sports": ["sport", "sports", "player", "game", "match", "athlete", "ball", "stadium"],
    "illustration": ["illustration", "drawing", "painting", "artwork", "cartoon"],
    "abstract": ["abstract", "pattern", "texture", "shapes", "gradient", "geometric"],
    "document": ["document", "screenshot", "text", "page", "form", "receipt"],
    "graphic": ["logo", "poster", "graphic", "banner", "icon", "design"],
    "aerial": ["aerial", "overhead", "drone", "bird's-eye"],
    "night_sky": ["stars", "night sky", "galaxy", "milky way", "moon"],
    "underwater": ["underwater", "ocean floor", "coral", "fish", "diver"],
}
