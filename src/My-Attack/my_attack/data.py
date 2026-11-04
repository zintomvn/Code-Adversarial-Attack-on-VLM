from .common import *
from .config_schema import cfg

class ImagePairDataset:
    def __init__(self, root, source_dir=None, target_dir=None, manifest=None):
        self.root = Path(root)
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.manifest = manifest
        self.transform = transforms.Compose([
            transforms.Resize(cfg.IMAGE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(cfg.IMAGE_SIZE),
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
