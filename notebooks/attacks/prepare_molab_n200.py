"""Package 200 original image pairs in the notebook's ImageFolder sort order."""

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile, ZIP_DEFLATED

from PIL import Image


def prepare(source_zip, destination, count=200):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
    rows, hashes, image_info = [], {}, {}
    with ZipFile(source_zip) as source:
        names = source.namelist()
        sources = sorted(n for n in names if '/bigscale/' in n and PurePosixPath(n).suffix.lower() in extensions)
        targets = sorted(n for n in names if '/target_images/' in n and PurePosixPath(n).suffix.lower() in extensions)
        if len(sources) != len(targets) or len(sources) < count:
            raise ValueError(f'Invalid source/target counts: {len(sources)}/{len(targets)}; requested {count}')
        target = destination / f'attack_data_n{count}.zip'
        with ZipFile(target, 'w', ZIP_DEFLATED) as output:
            for index, (src, tgt) in enumerate(zip(sources[:count], targets[:count])):
                row = {'index': index, 'source': src, 'target': tgt}
                for name in (src, tgt):
                    data = source.read(name)
                    with Image.open(io.BytesIO(data)) as image:
                        image_info[name] = {'size': list(image.size), 'mode': image.mode, 'format': image.format}
                        image.verify()
                    hashes[name] = hashlib.sha256(data).hexdigest()
                    output.writestr(name, data)  # Original bytes, without preprocessing/re-encoding.
                rows.append(row)
            manifest = ''.join(json.dumps(row) + '\n' for row in rows)
            output.writestr('pairs_200.jsonl', manifest)
            provenance = {
                'source_archive_name': Path(source_zip).name,
                'source_archive_sha256': hashlib.sha256(Path(source_zip).read_bytes()).hexdigest(),
                'available_pairs': len(sources), 'selected_pairs': count,
                'selection': 'First N pairs in sorted ImageFolder relative-path order; no outcome-based filtering.',
                'preprocessing': 'None during packaging; original image bytes preserved.',
                'sha256': hashes, 'images': image_info,
            }
            output.writestr('data_provenance.json', json.dumps(provenance, indent=2))
        (destination / 'pairs_200.jsonl').write_text(manifest, encoding='utf-8')
        (destination / 'data_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')
    with ZipFile(target) as check:
        if check.testzip() is not None:
            raise ValueError('ZIP CRC validation failed.')
        for name, digest in hashes.items():
            if hashlib.sha256(check.read(name)).hexdigest() != digest:
                raise ValueError(f'Packaged image changed: {name}')
    report = {'path': str(target.resolve()), 'pairs': len(rows), 'images': len(hashes),
              'bytes': target.stat().st_size, 'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
              'unique_source_images': len({hashes[r['source']] for r in rows}),
              'unique_target_images': len({hashes[r['target']] for r in rows})}
    (destination / 'data_validation.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-zip', type=Path, default=Path('src/FOA-Attack/resources/blackbox_dataset.zip'))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/molab_hypotheses_n200'))
    args = parser.parse_args()
    print(json.dumps(prepare(args.source_zip, args.output_dir), indent=2))
