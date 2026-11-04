from .common import *
from .config_schema import cfg

class V53Store:
    @staticmethod
    def write(path, value):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)

    @staticmethod
    def read(path):
        return json.loads(Path(path).read_text(encoding="utf-8"))

    @staticmethod
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    @staticmethod
    def fingerprint(record):
        return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()

    @staticmethod
    def config():
        ignored = {"V53_STAGE", "V53_APPROVAL", "V53_APPROVED_CONFIG_SHA256"}
        record = {name: str(value) if isinstance(value, Path) else value
                  for name, value in vars(cfg).items()
                  if name.startswith("V53_") and name not in ignored and not callable(value)}
        record["runtime_code_sha256"] = V53Store.code_hash()
        return json.loads(json.dumps(record))

    @staticmethod
    def code_hash():
        digest = hashlib.sha256()
        root = Path(__file__).parent
        for path in sorted(root.rglob("*.py")):
            digest.update(str(path.relative_to(root)).replace("\\", "/").encode())
            digest.update(path.read_bytes())
        return digest.hexdigest()

    @staticmethod
    def require_review(folder, key):
        path = Path(folder) / "reviews.json"
        if not path.exists():
            V53Store.write(path, {})
        review = V53Store.read(path).get(key, {})
        if review.get("status") != "PASS" or not review.get("evidence"):
            raise RuntimeError(f"REVIEW_PENDING: {path} -> {key}; record PASS + evidence after inspection")

    @staticmethod
    def require_checklist(folder):
        path = Path(folder)/"first_sample_checklist.json"
        names = ["augmented_data", "generated_concepts", "negative_data", "loss",
                 "attack_cropping", "attack_outcome", "adversarial_captions"]
        if not path.exists():
            V53Store.write(path, {name: {"status": "PENDING", "evidence": []} for name in names})
        records = V53Store.read(path)
        for name in names:
            record = records.get(name, {})
            if record.get("status") != "PASS" or not record.get("evidence"):
                raise RuntimeError(f"CHECKLIST_PENDING: {path} -> {name}")
        V53Store.require_review(folder, "all_checks")

    @staticmethod
    def seal(folder, names):
        path = Path(folder)/"integrity.json"
        hashes = {name: V53Store.sha(Path(folder)/name) for name in names}
        if path.exists() and V53Store.read(path) != hashes:
            raise RuntimeError(f"Cached stage changed: {path}")
        V53Store.write(path, hashes)

    @staticmethod
    def verify_seal(folder):
        path = Path(folder)/"integrity.json"
        if not path.exists():
            raise RuntimeError(f"Missing stage integrity: {path}")
        for name, digest in V53Store.read(path).items():
            if V53Store.sha(Path(folder)/name) != digest:
                raise RuntimeError(f"Stage artifact changed: {name}")

    @staticmethod
    def bundle(folder):
        folder = Path(folder)
        V53Store.write(folder / "sheet_pending_update.json", {
            "status": "PENDING_SYNC", "run_id": folder.name,
            "note": "Sync and verify research Sheet after every partial/completed run; preserve baseline cells.",
            "artifacts": str(folder), "official_asr": None,
            "gpt_score_metrics": V53Store.read(folder/"metrics.json") if (folder/"metrics.json").exists() else None,
            "compute": V53Store.read(folder/"budget.json") if (folder/"budget.json").exists() else None,
            "run_state": V53Store.read(folder/"run_status.json") if (folder/"run_status.json").exists() else None,
        })
        return shutil.make_archive(str(folder) + "-bundle", "zip", root_dir=folder)



class V53Heartbeat:
    def __init__(self, folder, stage, interval=30):
        self.folder = Path(folder)
        self.stage = stage
        self.interval = interval
        self.state = {}
        self.stop_event = threading.Event()

    def update(self, **state):
        self.state = {**self.state, **state}

    def _emit(self):
        row = {"stage": self.stage, "elapsed_seconds": round(time.monotonic()-self.started, 1), **self.state}
        print("[v5.3] " + json.dumps(row), flush=True)
        V53Store.write(self.folder / "heartbeat.json", row)

    def _loop(self):
        while not self.stop_event.wait(self.interval):
            self._emit()

    def __enter__(self):
        self.started = time.monotonic()
        self._emit()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop_event.set()
        self.thread.join()
        self._emit()



class V53Budget:
    def __init__(self, folder):
        self.path = Path(folder) / "budget.json"
        self.state = V53Store.read(self.path) if self.path.exists() else {
            "active_seconds": 0., "api_calls": 0, "reserved_usd": 0., "api_actual_usd": 0.}
        self.started = time.monotonic()
        self.previous = self.state["active_seconds"]

    def check(self):
        if self.state.get("blocked"):
            raise RuntimeError("Budget violation requires review; cached responses cannot bypass it")
        elapsed = self.previous + time.monotonic() - self.started
        self.state["active_seconds"] = elapsed
        V53Store.write(self.path, self.state)
        if elapsed >= self.time_limit():
            raise RuntimeError("COMPUTE_BUDGET_EXHAUSTED")

    def reserve_api(self):
        self.check()
        reserve = (cfg.V53_API_INPUT_TOKEN_CAP*cfg.V53_API_INPUT_USD_PER_M
                   + cfg.V53_API_OUTPUT_TOKENS*cfg.V53_API_OUTPUT_USD_PER_M)/1e6
        if (self.state["api_calls"] >= cfg.V53_API_MAX_CALLS
                or (cfg.V53_MAX_API_USD is not None and self.state["reserved_usd"] + reserve > cfg.V53_MAX_API_USD)):
            raise RuntimeError("API_BUDGET_EXHAUSTED")
        self.state["api_calls"] += 1
        self.state["reserved_usd"] += reserve
        V53Store.write(self.path, self.state)

    @staticmethod
    def time_limit():
        limits = [x for x in [cfg.V53_MAX_WALL_SECONDS,
                  cfg.V53_MAX_GPU_HOURS * 3600 if cfg.V53_MAX_GPU_HOURS is not None else None]
                  if x is not None]
        return min(limits) if limits else float("inf")

    def remaining_seconds(self):
        return self.time_limit() - (
            self.previous+time.monotonic()-self.started)

    def record_usage(self, usage):
        cost = (usage.get("input_tokens", 0)*cfg.V53_API_INPUT_USD_PER_M
                + usage.get("output_tokens", 0)*cfg.V53_API_OUTPUT_USD_PER_M)/1e6
        self.state["api_actual_usd"] += cost
        V53Store.write(self.path, self.state)
        if usage.get("input_tokens", 0) > cfg.V53_API_INPUT_TOKEN_CAP:
            self.state["blocked"] = "API_INPUT_RESERVATION_EXCEEDED"
            V53Store.write(self.path, self.state)
            raise RuntimeError("API_INPUT_RESERVATION_EXCEEDED: stop further calls")



class V53Pixels:
    @staticmethod
    def tensor(image):
        return torch.from_numpy(np.asarray(image.convert("RGB")).copy()).permute(2, 0, 1)[None].float()

    @staticmethod
    def project(source, delta, epsilon):
        return (source + delta.clamp(-epsilon, epsilon)).clamp(0, 255)

    @staticmethod
    def save(tensor, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        pixels = tensor.detach().cpu()[0].permute(1, 2, 0).clamp(0, 255).round().byte().numpy()
        Image.fromarray(pixels).save(path)

    @staticmethod
    def verify(path, source, epsilon):
        with Image.open(path) as handle:
            saved = V53Pixels.tensor(handle)
        difference = (saved-source.detach().cpu()).abs().max().item()
        if difference > epsilon + 1e-6:
            raise AssertionError(f"Saved PNG epsilon violation: {difference}")
        return {"linf_pixels": difference, "epsilon_pixels": epsilon,
                "png_sha256": V53Store.sha(path), "status": "PASS"}
