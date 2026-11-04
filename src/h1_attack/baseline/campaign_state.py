"""Atomic batch checkpoints and RNG restoration outside the author algorithms."""
import json, os, platform, shutil, time, zipfile
from pathlib import Path
from importlib.metadata import version
from baseline_runtime import read_json, write_json, sha, extract
from author_worker import rng_state, restore_rng
from campaign_runtime import Drive

class BatchLimitReached(Exception):pass

class State:
    def __init__(self,root,method,max_batches=None):
        self.root=Path(root);self.method=method;self.cfg=read_json(self.root/'config.json')
        self.rows=read_json(self.root/'selected_pairs.json');self.by_stem={r['stem']:r for r in self.rows}
        self.run_id=read_json(self.root/'identity.json')['fingerprint']
        self.drive=Drive(self.cfg['drive_parent_id']) if self.cfg['backup_drive'] else None
        self.remote=self.cfg['drive_prefix'].strip('/')+'/'+self.cfg['run_name']
        self.samples=self.root/'results'/method/'samples';self.samples.mkdir(parents=True,exist_ok=True)
        self.done={};self.resume_rng=None;self.max_batches=max_batches;self.new_batches=0
        import torch
        self.environment={'python':platform.python_version(),'gpu':torch.cuda.get_device_name(0),'cuda':torch.version.cuda,
          'packages':{p:version(p) for p in ('torch','torchvision','transformers','numpy','pillow')}}
        for marker in self.samples.glob('*/complete.json'):
            record=read_json(marker)
            if record['run_id']!=self.run_id or record['environment']!=self.environment:raise ValueError('Checkpoint identity/environment mismatch')
            for name,digest in record['artifacts_sha256'].items():
                if sha(marker.parent/name)!=digest:raise ValueError('Corrupt sample artifact')
            self.done[record['stem']]=record
        pending=False
        for r in self.rows:
            if r['stem'] in self.done:
                if pending:raise ValueError('Completed samples must be a contiguous prefix')
            else:pending=True
        commits=sorted((self.root/'batch_commits'/method).glob('*.commit.json'))
        if commits:
            record=read_json(commits[-1])
            if record['run_id']!=self.run_id or sha(self.root/record['rng_path'])!=record['rng_sha256']:raise ValueError('Corrupt RNG checkpoint')
            self.resume_rng=read_json(self.root/record['rng_path'])
        # Retry an upload which may have committed locally before a network failure.
        if self.drive:
            for archive in sorted((self.root/'checkpoints'/method).glob('*.zip')):
                self.drive.put(archive,self.remote+'/checkpoints/'+method+'/'+archive.name)

    def before(self,restore=True):
        if time.time()>=float(os.environ.get('BASELINE_DEADLINE','inf')):raise SystemExit(75)
        if self.max_batches is not None and self.new_batches>=self.max_batches:raise BatchLimitReached()
        if restore:self.restore()

    def restore(self):
        if self.resume_rng is not None:
            restore_rng(self.resume_rng);self.resume_rng=None
            print('RNG_RESTORED_AT_BATCH_BOUNDARY',flush=True)

    def commit(self,items):
        """Commit an entire author batch, including all its samples, in one ZIP."""
        import torch, torchvision
        import numpy as np
        from PIL import Image
        rng=rng_state();batch_num=len(list((self.root/'batch_commits'/self.method).glob('*.commit.json')))
        staging=self.root/'staging'/self.method/f'batch_{batch_num:05d}'
        if staging.exists():shutil.rmtree(staging)
        staging.mkdir(parents=True)
        destinations=[]
        for stem,source,target,adversarial in items:
            row=self.by_stem[stem];relative=Path('results')/self.method/'samples'/f'{row["index"]:05d}'
            sample=staging/relative;sample.mkdir(parents=True)
            for name,value in [('source.png',source),('target.png',target),('adversarial.png',adversarial)]:
                out=sample/name
                if isinstance(value,(str,Path)):shutil.copyfile(value,out)
                elif torch.is_tensor(value):torchvision.utils.save_image(value.detach().cpu(),out)
                else:value.save(out)
            with Image.open(sample/'source.png') as a,Image.open(sample/'adversarial.png') as b:
                linf=int(np.abs(np.asarray(a,dtype=np.int16)-np.asarray(b,dtype=np.int16)).max())
            if linf>self.cfg['epsilon']:raise ValueError('Saved perturbation exceeds epsilon: '+str(linf))
            record={'index':row['index'],'stem':stem,'run_id':self.run_id,'environment':self.environment,'saved_png_linf_0_255':linf,
              'source_input_sha256':row['source_sha256'],'target_input_sha256':row['target_sha256'],
              'artifacts_sha256':{p.name:sha(p) for p in sample.iterdir()}}
            write_json(sample/'complete.json',record);destinations.append(record)
        commit_dir=Path('batch_commits')/self.method
        rng_path=commit_dir/f'{batch_num:05d}.rng.json'
        write_json(staging/rng_path,rng)
        write_json(staging/commit_dir/f'{batch_num:05d}.commit.json',{'run_id':self.run_id,'rng_path':rng_path.as_posix(),'rng_sha256':sha(staging/rng_path),'stems':[r['stem'] for r in destinations]})
        archive=self.root/'checkpoints'/self.method/f'{batch_num:05d}.zip';archive.parent.mkdir(parents=True,exist_ok=True)
        temp=archive.with_suffix('.tmp')
        with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
            for p in staging.rglob('*'):
                if p.is_file():z.write(p,p.relative_to(staging).as_posix())
        temp.replace(archive)
        # Publish all local batch artifacts before reporting success or advancing RNG.
        extract(archive,self.root)
        if self.drive:self.drive.put(archive,self.remote+'/checkpoints/'+self.method+'/'+archive.name)
        for record in destinations:self.done[record['stem']]=record
        self.new_batches+=1;self.manifest()
        print('COMMITTED',self.method,len(self.done),'/',len(self.rows),'batch',batch_num,flush=True)
        restore_rng(rng)

    def manifest(self):
        from baseline_runtime import rebuild_manifests
        rebuild_manifests(self.root,{'methods':[self.method]},self.run_id)
        path=self.samples.parent/'manifest.jsonl'
        if self.drive:self.drive.put(path,self.remote+'/results/'+self.method+'/manifest.jsonl')
