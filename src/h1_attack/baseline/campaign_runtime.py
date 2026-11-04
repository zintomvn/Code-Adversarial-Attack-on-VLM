"""Portable campaign scheduling; original attack/loss implementations are immutable."""
from __future__ import annotations
import concurrent.futures, functools, hashlib, http.client, json, os, shutil, ssl, subprocess, sys, time, urllib.request, zipfile
from pathlib import Path
import baseline_runtime as legacy
from baseline_runtime import sha, read_json, write_json, extract, verify_sources, credentials, stage

REGISTRY={
 'o_attack':{'python':'3.10','torch':'2.7.0','vision':'0.22.0','transformers':'4.57.0','tokenizers':'0.22.1','environment':'om_v1_o_attack','adapter':'run_o_attack','author_config':'configs/attack/default.yaml'},
 'm_attack_v2':{'python':'3.10','torch':'2.10.0','vision':'0.25.0','transformers':'4.51.3','tokenizers':'0.21.4','environment':'om_v2_m_attack_v2_py310_tf4513','adapter':'run_m_attack','author_config':'config/ensemble_3models.yaml'}}

class Drive(legacy.Drive):
    """Retry only recoverable storage errors; keep original ID/MD5 guarantees."""
    def _retry(self, operation, *args):
        for attempt in range(4):
            try:return operation(*args)
            except (TimeoutError,ConnectionError,ssl.SSLError,http.client.RemoteDisconnected):
                if attempt==3:raise
                print('DRIVE_RETRY',operation.__name__,attempt+1,flush=True)
                time.sleep(5*2**attempt)
                import httplib2
                from google_auth_httplib2 import AuthorizedHttp
                from googleapiclient.discovery import build
                self.api=build('drive','v3',http=AuthorizedHttp(credentials(),http=httplib2.Http(timeout=180)),cache_discovery=False)
    def children(self,parent):return self._retry(super().children,parent)
    def put(self,local,remote):return self._retry(super().put,local,remote)
    def download(self,file_id,destination,md5=None):return self._retry(super().download,file_id,destination,md5)

def install_storage():
    # eval_worker imports this same legacy module after this hook is installed.
    legacy.Drive=Drive

def download_url(url,dest,checksum=None):
    dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists() and (checksum is None or sha(dest)==checksum):return dest
    temp=dest.with_suffix(dest.suffix+'.partial')
    for attempt in range(4):
        try:
            size=temp.stat().st_size if temp.exists() else 0
            request=urllib.request.Request(url,headers={'Range':f'bytes={size}-'} if size else {})
            with urllib.request.urlopen(request,timeout=180) as response:
                append=size>0 and response.status==206
                with temp.open('ab' if append else 'wb') as out:
                    shutil.copyfileobj(response,out,4<<20)
            if checksum and sha(temp)!=checksum:raise ValueError('Downloaded asset checksum mismatch: '+dest.name)
            temp.replace(dest);return dest
        except (TimeoutError,ConnectionError,OSError):
            if attempt==3:raise
            time.sleep(5*2**attempt)

def validate(cfg):
    if not cfg['methods'] or len(set(cfg['methods']))!=len(cfg['methods']) or set(cfg['methods'])-REGISTRY.keys():raise ValueError('Unknown or duplicate method')
    if Path(cfg['run_name']).name!=cfg['run_name'] or cfg['run_name'] in ('.','..'):raise ValueError('Invalid run_name')
    if not 0<cfg['n']<=1000 or cfg['steps']<1 or not 0<=cfg['epsilon']<=255:raise ValueError('Invalid n/steps/epsilon')
    for method in cfg['methods']:
        s=cfg['method_settings'][method]
        if s['batch_size']<1 or s['alpha']<=0:raise ValueError('Invalid method settings')
        protected=('data.','output.','model.device','model.device_ids','optim.steps','optim.epsilon','optim.alpha')
        if any(k.startswith(protected) for k in s['overrides']):raise ValueError('Use the main config fields for paths/device/budget')

def prepare(root,cfg,drive):
    path=Path(cfg['data_archive']) if cfg.get('data_archive') else root/'attack_data_n200.zip'
    if not path.exists():
        if not drive:raise FileNotFoundError('Configure data_archive or Drive data file ID')
        drive.download(cfg['data_drive_file_id'],path)
    if sha(path)!=cfg['data_sha256']:raise ValueError('Dataset SHA mismatch')
    extract(path,root/'data')
    rows=[json.loads(x) for x in (root/'data/pairs_200.jsonl').read_text().splitlines() if x.strip()][:cfg['n']]
    if len(rows)!=cfg['n']:raise ValueError('Insufficient dataset pairs')
    records=[]
    from PIL import Image
    for row in rows:
        stem=Path(row['source']).stem
        if stem!=Path(row['target']).stem:raise ValueError('Source/target IDs differ')
        record=dict(row,stem=stem)
        for role in ('source','target'):
            source=(root/'data'/row[role]).resolve()
            if not source.is_relative_to((root/'data').resolve()):raise ValueError('Unsafe dataset path')
            with Image.open(source) as image:image.verify()
            record[role+'_sha256']=sha(source)
            dest=root/'inputs'/role/'images'/source.name
            dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists() and sha(dest)!=sha(source):raise ValueError('Dataset input changed')
            if not dest.exists():shutil.copyfile(source,dest)
        records.append(record)
    if len({r['stem'] for r in records})!=len(records):raise ValueError('Duplicate sample ID')
    records.sort(key=lambda r:Path(r['source']).name)
    write_json(root/'selected_pairs.json',records)
    # Both original O-Attack caption banks are required, with no fabricated fallback.
    for role,folder in [('source','bigscale_1000/nips17'),('target','target_images_1000/1')]:
        p=root/'author/o_attack/data/text'/folder/'caption.json'
        bank={r['image']:r['caption'] for r in read_json(p)}
        for r in records:
            if not bank.get(Path(r[role]).name):raise ValueError('Missing original O-Attack caption: '+r['stem'])
    return records

def identity(root,cfg):
    keys=('n','epsilon','steps','methods','method_settings','model_revisions','data_sha256','retrieval')
    record={'config':{k:cfg[k] for k in keys},'sources':verify_sources(root),
      'pairs':read_json(root/'selected_pairs.json'),
      'runner':{p.name:sha(p) for p in root.glob('*.py')}}
    # Account-specific locations are not attack parameters.
    record['config']['retrieval']={k:v for k,v in cfg['retrieval'].items() if k!='reference_root'}
    value=hashlib.sha256(json.dumps(record,sort_keys=True).encode()).hexdigest()
    return {'fingerprint':value,'identity':record}

def interpreter(root,method):
    if method=='evaluation':return legacy.interpreter(root,'evaluation')
    spec=REGISTRY[method];env=root.parent/'environments'/spec['environment'];exe=env/'bin/python'
    stamp=env/'campaign-ready.json'
    if stamp.exists():return exe
    uv=shutil.which('uv')
    if not uv:raise RuntimeError('Install uv before running this campaign')
    run=lambda args:subprocess.run(list(map(str,args)),check=True)
    run([uv,'python','install',spec['python']])
    if not exe.exists():run([uv,'venv','--python',spec['python'],env])
    run([uv,'pip','install','--python',exe,'torch=='+spec['torch'],'torchvision=='+spec['vision'],'--index-url','https://download.pytorch.org/whl/cu128'])
    packages=['omegaconf==2.3.0','pillow==11.0.0','transformers=='+spec['transformers'],'tokenizers=='+spec['tokenizers'],'huggingface-hub==0.35.3','accelerate==1.12.0','timm==1.0.22','einops==0.8.1','PyYAML==6.0.3','tqdm==4.67.1','google-api-python-client','pydata-google-auth','python-dotenv']
    packages+=['numpy==2.0.2','matplotlib==3.10.7'] if method=='o_attack' else ['numpy==2.2.4','hydra-core==1.3.2','pytorch-lightning==2.6.1','torchmetrics==1.8.2','wandb==0.24.2','kornia==0.8.2','open_clip_torch==3.2.0','scipy==1.15.3']
    run([uv,'pip','install','--python',exe,*packages])
    freeze=subprocess.check_output([uv,'pip','freeze','--python',str(exe)],text=True)
    write_json(stamp,{'spec':spec,'packages':freeze.splitlines()});return exe

def count(root,method):return len(list((root/'results'/method/'samples').glob('*/complete.json')))

def controller(root,cfg):
    import fcntl
    root=Path(root).resolve();validate(cfg);install_storage()
    with (root/'controller.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        drive=Drive(cfg['drive_parent_id']) if cfg['backup_drive'] else None
        remote=cfg['drive_prefix'].strip('/')+'/'+cfg['run_name']
        stage(root,'preparing_data');pairs=prepare(root,cfg,drive);ident=identity(root,cfg)
        path=root/'identity.json'
        if path.exists() and read_json(path)['fingerprint']!=ident['fingerprint']:raise ValueError('Run identity changed; choose a new run_name')
        write_json(path,ident)
        if drive:
            folder=drive.folder(remote);old=drive.find(folder,'identity.json')
            if old:
                dest=root/'remote_identity.json';drive.download(old['id'],dest,old.get('md5Checksum'))
                if read_json(dest)['fingerprint']!=ident['fingerprint']:raise ValueError('Drive run identity mismatch')
            drive.put(path,remote+'/identity.json')
            drive.put(root/'config.json',remote+'/config.json')
            for method in cfg['methods']:
                drive.restore_tree(remote+'/checkpoints/'+method,root/'checkpoints'/method)
                for z in sorted((root/'checkpoints'/method).glob('*.zip')):extract(z,root)
        env=os.environ.copy();env.update(PYTHONPATH=str(root)+os.pathsep+env.get('PYTHONPATH',''),WANDB_MODE='disabled',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS=str(cfg['cpu_threads']),MKL_NUM_THREADS=str(cfg['cpu_threads']),HF_HOME=str(root.parent/'hf_cache'),HF_HUB_DISABLE_XET='1',BASELINE_DEADLINE=str(time.time()+cfg['session_hours']*3600))
        phases=[True,False] if cfg.get('verify_first_batch') else [False]
        for verify in phases:
            for method in cfg['methods']:
                if count(root,method)==cfg['n']:continue
                if verify and count(root,method)>0:continue
                stage(root,'installing',method=method);exe=interpreter(root,method)
                stage(root,'attacking',method=method,verification_batch=verify)
                command=[str(exe),'-u',str(root/'campaign_worker.py'),str(root),method]
                if verify:command+=['--max-new-batches','1']
                with (root/'logs'/f'{method}.log').open('a') as log:
                    code=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
                if code:
                    stage(root,'paused_session' if code==75 else 'attack_failed',method=method,exit_code=code);return code
        legacy.rebuild_manifests(root,cfg,ident['fingerprint'])
        pending=[]
        if cfg['run_evaluation']:
            exe=interpreter(root,'evaluation')
            for model in cfg['evaluation']['models']:
                stage(root,'evaluating',model=model)
                with (root/'logs/evaluation.log').open('a') as log:
                    code=subprocess.run([str(exe),'-u',str(root/'campaign_worker.py'),str(root),'evaluation','--model',model],env=env,stdout=log,stderr=subprocess.STDOUT).returncode
                if code==75:stage(root,'paused_session',model=model);return code
                if code:pending.append({'model':model,'exit_code':code})
        stage(root,'evaluation_pending' if pending else 'complete',pending_models=pending)
        if drive:drive.put(root/'status.json',remote+'/status.json')
        return 76 if pending else 0

def main():
    root=Path(sys.argv[1]).resolve()
    try:return controller(root,read_json(root/'config.json'))
    except Exception as exc:
        stage(root,'failed',error_type=type(exc).__name__,detail=str(exc)[:300] if isinstance(exc,(ValueError,FileNotFoundError)) else 'Inspect process log')
        raise

if __name__=='__main__':sys.exit(main())
