"""Verify saved attack artifacts and build pixel-deduplicated caption inputs."""
import hashlib
import io
import json
import re
import zipfile

import numpy as np
from PIL import Image


def _image(data):
    image = Image.open(io.BytesIO(data)).convert('RGB')
    pixels = np.array(image)
    return (pixels, hashlib.sha256(str(pixels.shape).encode() + pixels.tobytes()).hexdigest())


def _zip_bytes(files):
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w') as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    return output.getvalue()


def build_caption_bundle(results, epsilon=16):
    if not results:
        raise ValueError('No attack artifacts')
    archives = {}
    cohort = None
    try:
        for variant, result in results.items():
            if not re.fullmatch('[A-Za-z0-9_-]+', variant):
                raise ValueError('Unsafe variant name')
            summary = result['summary']
            if summary['state'] != 'completed_verified':
                raise ValueError('Attack incomplete')
            data = result['archive']
            if hashlib.sha256(data).hexdigest() != summary['archive_sha256']:
                raise ValueError('Attack archive hash mismatch')
            archive = zipfile.ZipFile(io.BytesIO(data))
            archives[variant] = archive
            pairs = {str(r['pair']): r for r in summary['samples']}
            if not pairs or len(pairs) != len(summary['samples']):
                raise ValueError('Empty or duplicate cohort')
            if cohort is None:
                cohort = set(pairs)
            elif cohort != set(pairs):
                raise ValueError('Variants have different cohorts')
        files = {}
        rows = []
        by_pixel = {}
        logical = []
        first = next(iter(results))

        def add(pair, variant, role, data, source_path=None):
            pixels, identity = _image(data)
            alias = {'pair_id': int(pair) if pair.isdecimal() else pair, 'role': role, 'variant': variant, 'original_file_sha256': hashlib.sha256(data).hexdigest()}
            if identity in by_pixel:
                row = by_pixel[identity]
                row['aliases'].append(alias)
            else:
                relative = f'images/{pair}_{variant}_{role}.png'
                files[relative] = data
                row = {'id': f'image_{len(rows):05d}', 'path': relative, 'file_sha256': hashlib.sha256(data).hexdigest(), 'pixel_sha256': identity, 'role': role, 'aliases': [alias]}
                if source_path is not None:
                    row['source_path'] = source_path
                rows.append(row)
                by_pixel[identity] = row
            logical.append({'image_id': row['id'], **alias})
            return (row['path'], pixels)
        for pair in sorted(cohort):
            if not re.fullmatch('[A-Za-z0-9_-]+', pair):
                raise ValueError('Unsafe pair ID')
            prefix = f'results/m_attack_v2/samples/{pair}/'
            source = archives[first].read(first + '/' + prefix + 'source.png')
            target = archives[first].read(first + '/' + prefix + 'target.png')
            source_path, source_pixels = add(pair, 'shared', 'source', source)
            _, target_pixels = add(pair, 'shared', 'target', target)
            for variant, result in results.items():
                archive = archives[variant]
                base = variant + '/' + prefix
                archive.read(base + 'complete.json')
                if not np.array_equal(source_pixels, _image(archive.read(base + 'source.png'))[0]):
                    raise ValueError('Source mismatch')
                if not np.array_equal(target_pixels, _image(archive.read(base + 'target.png'))[0]):
                    raise ValueError('Target mismatch')
                data = archive.read(base + 'adversarial.png')
                expected = next((r['output_sha256'] for r in result['summary']['samples'] if str(r['pair']) == pair))
                if hashlib.sha256(data).hexdigest() != expected:
                    raise ValueError('Adversarial PNG hash mismatch')
                pixels, _ = _image(data)
                if pixels.shape != source_pixels.shape or np.abs(pixels.astype('int16') - source_pixels.astype('int16')).max() > epsilon:
                    raise ValueError('Invalid adversarial shape/epsilon')
                add(pair, variant, 'adversarial', data, source_path)
        manifest = {'schema': 'exact-png-caption-tasks-v1', 'rows': rows, 'logical_tasks': logical, 'provenance': {'stage': 'automatic post-attack', 'variants': list(results), 'unique_images': len(rows), 'logical_slots': len(logical)}}
        files['manifest.json'] = json.dumps(manifest, sort_keys=True, indent=2).encode()
        archive = _zip_bytes(files)
        return {'archive': archive, 'sha256': hashlib.sha256(archive).hexdigest(), 'manifest': manifest}
    finally:
        for archive in archives.values():
            archive.close()
