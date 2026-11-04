"""Migrated directly from CARA v6.4.1; see SOURCE_MAP.json and ATTRIBUTION.md."""
from .config import INSTALL_PACKAGES, RUN_ATTACK, RUN_EVALUATION, RUN_VISUALIZATION, NUM_SAMPLES, START_INDEX, SEED, DATA_ROOT, SOURCE_DIR, TARGET_DIR, PAIR_MANIFEST, OUTPUT_DIR, K, STEPS, XAI_METHOD, EPSILON, ALPHA, MOMENTUM, TEMPERATURE, OMEGA, CROP_AREA, ALPHA_GLOBAL, ALPHA_CROP, SAVE_BRANCH_IMAGES, SAVE_EVERY, SAVE_CROP_EVERY, ATTENTION_LAYER, NEGATIVE_RETRIES, CAPTION_ATTEMPTS, IMAGE_SIZE, CLIP_G14_ID, CLIP_B16_ID, CLIP_B32_ID, QWEN_ID, EVALUATION_ID, CLIP_DEVICES, ATTACK_DEVICE, QWEN_MAX_MEMORY_GIB, IMAGE_WEIGHT, TEXT_WEIGHT, CAPTION_TEMPERATURE, CAPTION_TOP_P, EVALUATION_PROMPT, RUN_ID, MODEL_REVISIONS
from .common import *
class ImagePairDataset:
    def __init__(self, root, source_dir=None, target_dir=None, manifest=None):
        self.root = Path(root)
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.manifest = manifest
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(IMAGE_SIZE),
        ])

    def find_directory(self, name, override):
        if override is not None:
            directory = Path(override)
            if not directory.is_dir():
                raise FileNotFoundError(directory)
            return directory

        matches = [p for p in self.root.rglob(name) if p.is_dir()]
        if self.root.name == name:
            matches.insert(0, self.root)
        if len(matches) != 1:
            raise ValueError(f"Expected one {name} folder; found: {matches}")
        return matches[0]

    def get_pairs(self, start=0, count=5):
        if self.manifest is not None:
            table = pd.read_csv(self.manifest)
            if not {"source", "target"}.issubset(table.columns):
                raise ValueError("PAIR_MANIFEST needs source,target columns")
            base = Path(self.manifest).parent
            pairs = []
            for row in table.itertuples(index=False):
                source = Path(row.source)
                target = Path(row.target)
                pairs.append((source if source.is_absolute() else base / source,
                              target if target.is_absolute() else base / target))
        else:
            source_set = ImageFolder(self.find_directory("bigscale", self.source_dir))
            target_set = ImageFolder(self.find_directory("target_images", self.target_dir))
            if len(source_set) != len(target_set):
                raise ValueError("Source and target counts differ; use PAIR_MANIFEST")
            pairs = list(zip([row[0] for row in source_set.samples],
                             [row[0] for row in target_set.samples]))

        selected = pairs[start:start + count]
        if self.manifest is None:
            matches = sum(
                Path(source).stem.split("_")[0] == Path(target).stem.split("_")[0]
                for source, target in selected
            )
            if matches != len(selected):
                print("WARNING: File IDs may not match. Inspect the pair preview or supply PAIR_MANIFEST.")
        if len(selected) != count:
            raise ValueError(f"Requested {count} pairs, got {len(selected)}")
        for source, target in selected:
            if not Path(source).is_file() or not Path(target).is_file():
                raise FileNotFoundError(f"Missing source/target: {source}, {target}")
            if Path(source).resolve() == Path(target).resolve():
                raise ValueError("Source and target cannot be the same file")
        return selected

    def load_image(self, path):
        with Image.open(path) as handle:
            image = ImageOps.exif_transpose(handle).convert("RGB")
        return self.transform(image)

