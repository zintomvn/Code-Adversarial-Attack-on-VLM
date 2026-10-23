"""Source integrity, paired assets, atomic batch restore and RNG checks."""
import ast, hashlib, json, random, sys, tempfile, zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.append(str(HERE.parent))
from baseline_runtime import extract,read_json,write_json,verify_sources

def verify():
    checks={}
    with tempfile.TemporaryDirectory(prefix='om_verify_',dir=HERE) as tmp:
        root=Path(tmp).resolve();extract(HERE/'campaign_code.zip',root)
        sys.path.insert(0,str(root))
        from campaign_runtime import prepare,identity
        from campaign_state import State
        from retrieval_worker import extract_coco_archive
        from author_worker import rng_state,restore_rng
        import torch, numpy as np
        from unittest.mock import patch
        lock=verify_sources(root);checks['official_files_verified']=sum(len(s['files']) for s in lock.values())
        cfg=read_json(HERE/'default_config.json');cfg['backup_drive']=False;cfg['n']=2
        cfg['data_archive']=str(HERE.parents[3]/'outputs/molab_hypotheses_n200/attack_data_n200.zip')
        pairs=prepare(root,cfg,None);write_json(root/'config.json',cfg);write_json(root/'identity.json',identity(root,cfg))
        checks['original_caption_banks_cover_cohort']=True
        archive=root/'tiny_coco.zip';reference=root/'interrupted_coco';(reference/'train2014').mkdir(parents=True)
        with zipfile.ZipFile(archive,'w') as z:
            z.writestr('train2014/a.jpg',b'complete JPEG fixture')
            z.writestr('train2014/b.jpg',b'second JPEG fixture')
        (reference/'train2014/a.jpg').write_bytes(b'interrupted')
        extract_coco_archive(archive,reference,expected=2)
        assert (reference/'train2014/a.jpg').read_bytes()==b'complete JPEG fixture'
        assert (reference/'train2014/b.jpg').read_bytes()==b'second JPEG fixture'
        assert read_json(reference/'extraction_complete.json')['images']==2
        checks['interrupted_coco_extraction_recovers']=True
        def draw():return [random.random(),float(np.random.rand()),torch.rand(3).tolist()]
        with patch('torch.cuda.get_device_name',return_value='validation device'):
            state=State(root,'o_attack')
            random.seed(9);np.random.seed(9);torch.manual_seed(9)
            tensor=torch.ones(3,12,12)*0.3
            saved=rng_state()
            state.commit([(r['stem'],tensor,tensor,tensor) for r in pairs])
            first=draw();restore_rng(saved);assert draw()==first
            resumed=State(root,'o_attack');resumed.before();assert draw()==first
            checks['batch_rng_json_roundtrip']=True
            archives=list((root/'checkpoints/o_attack').glob('*.zip'));assert len(archives)==1
            restored=root/'fresh_account';restored.mkdir()
            for name in ('config.json','selected_pairs.json','identity.json'):(restored/name).write_bytes((root/name).read_bytes())
            extract(archives[0],restored)
            second=State(restored,'o_attack');assert len(second.done)==2
            checks['whole_batch_restores_in_fresh_root']=True
            artifact=next(second.samples.glob('*/adversarial.png'));artifact.write_bytes(b'corruption')
            try:State(restored,'o_attack')
            except ValueError:checks['corrupt_sample_rejected']=True
            else:raise AssertionError('Corruption accepted')
        for file in HERE.glob('*.py'):ast.parse(file.read_text(encoding='utf8'))
        for file in HERE.glob('*.ipynb'):
            for cell in read_json(file)['cells']:
                if cell['cell_type']=='code':ast.parse(''.join(cell['source']))
        checks['portable_notebooks_parse']=True
    write_json(HERE/'local_validation.json',checks);return checks

if __name__=='__main__':print(json.dumps(verify(),indent=2))
