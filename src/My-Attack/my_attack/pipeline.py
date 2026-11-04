"""Direct Python orchestration. No notebook or nbclient is used."""
from .common import *
from .config_schema import cfg
from .utils import V53Store, V53Pixels, V53Budget, V53Heartbeat
from .preparation import V53Preparation
from .surrogates.clip import V53CLIPEnsemble
from .attack import V53Attack
from .caption_bank import V53HFVision
from .evaluation import V53OpenAI, V53Evaluation
from .visualization import V53Preview
from .persistence import StagePersistence
from .drive_persistence import DriveStagePersistence


class MyAttackPipeline(V53Preparation):
    def __init__(self, cache_root=None, drive_transport=None):
        super().__init__()
        self.persistence = DriveStagePersistence(self.root, drive_transport) if drive_transport else StagePersistence(self.root, cache_root)
        self.budget = V53Budget(self.root)
        self.budget.durable_callback = self.snapshot

    def identity(self):
        if not cfg.V53_APPROVAL:
            raise RuntimeError('Explicit run authorization required')
        if not torch.cuda.is_available() or cfg.V53_EXPECTED_GPU not in torch.cuda.get_device_name(cfg.V53_DEVICE):
            raise RuntimeError('Expected the configured MoLab RTX6000 GPU')
        if not all(re.fullmatch(r'[0-9a-f]{40}', revision) for revision in cfg.V53_REVISIONS.values()):
            raise ValueError('Immutable Hugging Face model/processor revisions required')
        pairs = self.inspect_pairs()
        for pair in pairs:
            expected = cfg.V53_PAIR_HASHES[pair['id']]
            for role in ('source', 'target'):
                if pair[role + '_sha256'] != expected[role]:
                    raise RuntimeError('Dataset pair hash mismatch')
        record = {'config': V53Store.config(), 'pairs': pairs}
        path = self.root / 'identity.json'
        if path.exists() and V53Store.read(path) != record:
            raise RuntimeError('Run identity changed; use a new run ID')
        V53Store.write(path, record)
        V53Store.write(self.root / 'environment.json', {
            'python': platform.python_version(),
            'packages': {name: metadata.version(name) for name in
                         ['torch', 'torchvision', 'transformers', 'diffusers', 'openai', 'accelerate']},
            'gpu': torch.cuda.get_device_name(cfg.V53_DEVICE)})
        return pairs

    def snapshot(self, key, folder):
        if isinstance(self.persistence, DriveStagePersistence) and not self.persistence.healthy:
            raise RuntimeError('Drive storage failed; refusing further work')
        result = self.persistence.save(key, folder)
        print('[retained] ' + json.dumps(result), flush=True)

    def attack_pair(self, pair, heartbeat):
        shared = self.prepare(pair, self.budget, heartbeat)
        self.snapshot(f'{pair["id"]}_prepared', shared)
        # One ensemble load for both arms; no changes to features or gradients.
        encoder = None
        try:
            for variant in cfg.V53_VARIANTS:
                folder = self.root / pair['id'] / variant
                if (folder / 'adversarial.png').exists():
                    V53Store.verify_seal(folder)
                    continue
                if encoder is None:
                    encoder = V53CLIPEnsemble()
                with Image.open(shared / 'source.png') as image:
                    source = V53Pixels.tensor(image).to(cfg.V53_DEVICE)
                with Image.open(shared / 'effective_target.png') as image:
                    target = V53Pixels.tensor(image).to(cfg.V53_DEVICE)
                attack = V53Attack(encoder, self.budget, heartbeat, self.snapshot)
                heartbeat.update(stage='attack', sample=pair['id'], variant=variant)
                attack.run(source, target, V53Store.read(shared / 'bank.json'),
                           V53Store.read(shared / 'selection.json'), folder, pair['id'], variant)
                V53Store.write(folder / 'bounds.json', V53Pixels.verify(
                    folder / 'adversarial.png', source, cfg.V53_EPSILON))
                self.snapshot(f'{pair["id"]}_{variant}_attack', folder)
        finally:
            del encoder
            release_memory()

    def caption_pair(self, pair, heartbeat):
        shared = self.root / pair['id'] / 'shared'
        api = None
        for victim, model_id in cfg.V53_VICTIMS.items():
            adapter = None
            jobs = [('source', shared, shared / 'source.png'), ('target', shared, shared / 'target.png')]
            jobs += [('adversarial', self.root / pair['id'] / variant,
                      self.root / pair['id'] / variant / 'adversarial.png') for variant in cfg.V53_VARIANTS]
            try:
                for role, folder, image_path in jobs:
                    output = folder / 'captions' / f'{victim}_{role}.json'
                    self.budget.check()
                    heartbeat.update(sample=pair['id'], stage='caption', victim=victim, role=role)
                    if not image_path.exists():
                        raise FileNotFoundError(f'Attack output missing; caption-only stage will not attack: {image_path}')
                    if output.exists():
                        record = V53Store.read(output)
                        if (record.get('image_sha256') != V53Store.sha(image_path)
                                or record.get('model') != model_id or record.get('prompt') != cfg.V53_CAPTION_PROMPT):
                            raise RuntimeError('Caption identity mismatch')
                        if victim != 'gpt' and record.get('revision') != cfg.V53_REVISIONS[model_id]:
                            raise RuntimeError('Caption revision mismatch')
                    elif victim == 'gpt':
                        if api is None:
                            api = V53OpenAI(self.budget)
                        record = api.caption(image_path, output)
                    else:
                        if adapter is None:
                            adapter = V53HFVision(model_id)
                        with Image.open(image_path) as image:
                            record = adapter.caption(image.convert('RGB'), sample=False)
                        record['image_sha256'] = V53Store.sha(image_path)
                        V53Store.write(output, record)
                    if not record.get('text') or record.get('status', 'completed') != 'completed':
                        raise RuntimeError(f'Incomplete caption preserved: {output}')
                    if record.get('possible_truncation'):
                        raise RuntimeError(f'Truncated caption preserved: {output}')
                    self.snapshot(f'{pair["id"]}_{folder.name}_{victim}_{role}_caption', folder)
            finally:
                del adapter
                release_memory()

    def evaluate_pair(self, pair, heartbeat):
        shared = self.root / pair['id'] / 'shared'
        api = V53OpenAI(self.budget)
        for variant in cfg.V53_VARIANTS:
            folder = self.root / pair['id'] / variant
            rows = []
            for victim in cfg.V53_VICTIMS:
                heartbeat.update(sample=pair['id'], stage='gpt_score', variant=variant, victim=victim)
                source = V53Store.read(shared / 'captions' / f'{victim}_source.json')['text']
                target = V53Store.read(shared / 'captions' / f'{victim}_target.json')['text']
                adv = V53Store.read(folder / 'captions' / f'{victim}_adversarial.json')['text']
                score = api.score(target, adv, folder / 'scores' / f'{victim}.json')
                rows.append({'victim': victim, 'score': score['score'], 'success': score['success'],
                             'source_caption': source, 'target_caption': target, 'adversarial_caption': adv})
                self.snapshot(f'{pair["id"]}_{variant}_{victim}_score', folder)
            V53Store.write(folder / 'evaluation.json', {
                'status': 'COMPLETED', 'rows': rows, 'judge': cfg.V53_GPT_MODEL,
                'threshold': cfg.V53_ASR_THRESHOLD, 'official_asr': None,
                'protocol': 'v53_gpt_score_not_published_official_ASR'})
            V53Preview.show(shared, folder, step=cfg.V53_STEPS)
            self.snapshot(f'{pair["id"]}_{variant}_evaluated', folder)
        self.record_checks(pair)

    def record_checks(self, pair):
        folder = self.root / pair['id']
        checks = {
            'augmented_data': {'status': 'PASS', 'evidence': ['shared/integrity.json', 'shared/source.png', 'shared/target.png'],
                               'note': 'Hash/RGB/preprocessing checks only; semantic inspection waived by user.'},
            'generated_concepts': {'status': 'PENDING', 'evidence': ['shared/bank.json', 'shared/diffusion.json'],
                                   'note': 'Manual semantic grounding waived, never fabricated as PASS.'},
            'negative_data': {'status': 'PENDING', 'evidence': ['shared/bank.json', 'shared/selection.json'],
                              'note': 'Duplicates and exact overlap logged; semantic false negatives unreviewed.'},
            'loss': {'status': 'PASS', 'evidence': [f'{v}/history.jsonl' for v in cfg.V53_VARIANTS]},
            'attack_cropping': {'status': 'PASS', 'evidence': [f'{v}/views/step_001/record.json' for v in cfg.V53_VARIANTS],
                                'note': 'Area/coordinates and gradient reach checked automatically.'},
            'attack_outcome': {'status': 'PASS', 'evidence': [f'{v}/bounds.json' for v in cfg.V53_VARIANTS]
                               + [f'{v}/evaluation.json' for v in cfg.V53_VARIANTS],
                               'note': 'PASS means bounded PNG and completed configured judge; it does not mean attack success.'},
            'adversarial_captions': {'status': 'PENDING', 'evidence': [f'{v}/captions' for v in cfg.V53_VARIANTS],
                                     'note': 'Caption protocol/finite outputs checked; manual quality inspection waived.'}}
        V53Store.write(folder / 'first_sample_checklist.json', checks)
        V53Store.write(folder / 'review_waiver.json', {
            'authorization': cfg.V53_APPROVAL, 'manual_checks': 'WAIVED_BY_USER',
            'semantic_checks_complete': False})
        self.snapshot(f'{pair["id"]}_checklist', folder / 'shared')

    def summarize(self):
        all_rows = V53Evaluation.summarize(self.root)
        for label, ids in [('pilot', [x for x in cfg.V53_PAIR_IDS if x in ['1', '10']]),
                           ('confirmation', [x for x in cfg.V53_PAIR_IDS if x not in ['1', '10']])]:
            rows = []
            for variant in cfg.V53_VARIANTS:
                for victim in cfg.V53_VICTIMS:
                    scores = []
                    for sample in ids:
                        path = self.root / sample / variant / 'scores' / f'{victim}.json'
                        if path.exists():
                            scores.append(V53Store.read(path)['score'])
                    rows.append({'variant': variant, 'victim': victim, 'planned': len(ids), 'evaluated': len(scores),
                                 'successes': sum(s >= cfg.V53_ASR_THRESHOLD for s in scores),
                                 'gpt_score_asr': sum(s >= cfg.V53_ASR_THRESHOLD for s in scores) / len(scores)
                                 if scores and len(scores) == len(ids) else None, 'official_asr': None})
            V53Store.write(self.root / f'metrics_{label}.json', rows)
        return all_rows

    def run(self, stage='all', restore=False):
        if isinstance(self.persistence, DriveStagePersistence) and self.persistence.index['files'] and not restore:
            raise RuntimeError('Existing remote run: use --restore; never overwrite a committed run')
        if restore:
            print('Restored stages:', self.persistence.restore(), flush=True)
            if not isinstance(self.persistence, DriveStagePersistence):
                for path in (self.root / '_metadata').glob('*.json'):
                    shutil.copy2(path, self.root / path.name)
            self.budget = V53Budget(self.root)
            self.budget.durable_callback = self.snapshot
        pairs = self.identity()
        if isinstance(self.persistence, DriveStagePersistence):
            import io
            import zipfile
            project = Path(__file__).resolve().parent.parent
            source = io.BytesIO()
            with zipfile.ZipFile(source, 'w', zipfile.ZIP_DEFLATED) as archive:
                for path in sorted(project.rglob('*')):
                    relative = path.relative_to(project).as_posix()
                    if (path.is_file() and self.persistence.allowed(relative)
                            and path.suffix in {'.py', '.md', '.txt'}):
                        archive.write(path, relative)
                archive.writestr('resolved_config.json', json.dumps(V53Store.config(), indent=2))
            payload = source.getvalue()
            self.persistence.verified_put('code/source_' + self.persistence.sha(payload)[:16] + '.zip', payload)
            self.snapshot('identity_environment_before_inference', self.root)
        try:
            with V53Heartbeat(self.root, stage, cfg.V53_LOG_SECONDS) as heartbeat:
                for pair in pairs:
                    heartbeat.update(sample=pair['id'], stage='prepare' if stage in ['all', 'attack'] else stage)
                    if stage in ['all', 'attack']:
                        self.attack_pair(pair, heartbeat)
                    if stage in ['all', 'caption']:
                        self.caption_pair(pair, heartbeat)
                    if stage in ['all', 'evaluate']:
                        self.evaluate_pair(pair, heartbeat)
                    self.summarize()
                    V53Store.write(self.root / 'run_status.json', {
                        'status': 'RUNNING', 'last_completed_sample': pair['id'], 'stage': stage,
                        'manual_checks': 'WAIVED_BY_USER'})
                    self.snapshot(f'{pair["id"]}_complete', self.root / pair['id'])
        except BaseException as error:
            import traceback
            (self.root / 'error_traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
            V53Store.write(self.root / 'run_status.json', {
                'status': 'PARTIAL_FAILED', 'error': f'{type(error).__name__}: {error}', 'stage': stage})
            raise
        else:
            V53Store.write(self.root / 'run_status.json', {'status': 'COMPLETED', 'stage': stage,
                           'manual_checks': 'WAIVED_BY_USER'})
        finally:
            self.budget.state['active_seconds'] = self.budget.previous + time.monotonic() - self.budget.started
            V53Store.write(self.budget.path, self.budget.state)
            self.summarize()
            if not isinstance(self.persistence, DriveStagePersistence):
                V53Store.bundle(self.root)
            # Small run-wide metadata snapshot; never rearchive the growing batch here.
            meta = self.root / '_metadata'
            meta.mkdir(exist_ok=True)
            for path in self.root.glob('*.json'):
                shutil.copy2(path, meta / path.name)
            if getattr(self.persistence, 'healthy', True):
                self.snapshot('run_metadata_' + str(self.budget.state['api_calls']), meta)
