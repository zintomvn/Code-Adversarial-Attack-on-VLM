"""Freeze official sources and build small portable Molab/Jupyter notebooks."""
from pathlib import Path
import ast, base64, concurrent.futures, hashlib, io, json, pprint, subprocess, urllib.request, zipfile

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
PARENT=HERE.parent
SPECS={
 'o_attack':('O-Attack','Summu77/O-Attack','5585398e89a4d3415d0f29ff924368e8ba3c9cef'),
 'm_attack_v2':('M-Attack-V2','VILA-Lab/M-Attack-V2','8faca6d95e1caa190d676575489ab81bb31e4d03')}

def freeze():
    files={}; lock={}; jobs=[]; differences=[]
    previous=json.loads((HERE/'SOURCE_LOCK.json').read_text()) if (HERE/'SOURCE_LOCK.json').exists() else {}
    for method,(local,remote,commit) in SPECS.items():
        root=REPO/'src'/local
        paths=([*root.glob('oattack/**/*.py'),root/'configs/attack/default.yaml',root/'requirements.txt',
          *root.glob('data/text/**/*.json')] if method=='o_attack' else
          [*root.glob('attack/**/*.py'),*root.glob('surrogates/**/*.py'),root/'generate_ad_sample_parallel.py',
           root/'config_schema.py',root/'utils.py',root/'retrieval.py',root/'config/ensemble_3models.yaml',root/'pyproject.toml',root/'LICENSE'])
        if previous.get(method,{}).get('commit')==commit:
            paths=[root/rel for rel in previous[method]['files']]
            if method=='m_attack_v2':paths.append(root/'LICENSE')
        lock[method]={'repository':'https://github.com/'+remote,'commit':commit,'files':{}}
        for path in paths: jobs.append((method,root,path,remote,commit))
    cache=HERE/'verified_sources'
    def obtain(job):
        method,root,path,remote,commit=job; rel=path.relative_to(root).as_posix()
        cached=cache/method/rel
        if cached.exists(): data=cached.read_bytes()
        else:
            url=f'https://raw.githubusercontent.com/{remote}/{commit}/{rel}'
            for attempt in range(4):
                try:
                    with urllib.request.urlopen(url,timeout=90) as response: data=response.read()
                    break
                except Exception:
                    if attempt==3: raise
            cached.parent.mkdir(parents=True,exist_ok=True); cached.write_bytes(data)
        expected=previous.get(method,{}).get('files',{}).get(rel) if previous.get(method,{}).get('commit')==commit else None
        if expected and hashlib.sha256(data).hexdigest()!=expected:
            raise ValueError(f'Official cache differs from pinned checksum: {method}/{rel}')
        differs=not path.exists() or data.replace(b'\r\n',b'\n')!=path.read_bytes().replace(b'\r\n',b'\n')
        return method,rel,data,differs
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for method,rel,data,differs in pool.map(obtain,jobs):
            if differs:differences.append(method+'/'+rel)
            # The license travels with every embedded copy; algorithm identity is unchanged.
            if rel!='LICENSE':lock[method]['files'][rel]=hashlib.sha256(data).hexdigest()
            files['author/'+method+'/'+rel]=data
            if rel=='LICENSE':(HERE/'M_ATTACK_V2_LICENSE.txt').write_bytes(data)
    # Separate data assets, fetched by pinned URL and SHA, never embedded in notebook source.
    assets={}
    rows=[json.loads(s) for s in (REPO/'outputs/molab_hypotheses_n200/pairs_200.jsonl').read_text().splitlines()]
    for row in rows:
        stem=Path(row['source']).stem
        for rank in (1,2):
            rel=f'resources/retrieved_embeddings/{stem}/{rank}.jpg'
            p=REPO/'src/M-Attack-V2'/rel
            if p.exists(): assets[rel]=hashlib.sha256(p.read_bytes()).hexdigest()
    files['released_retrieval_assets.json']=(json.dumps(assets,sort_keys=True)+'\n').encode()
    files['author/SOURCE_LOCK.json']=(json.dumps(lock,indent=2)+'\n').encode()
    (HERE/'SOURCE_LOCK.json').write_bytes(files['author/SOURCE_LOCK.json'])
    (HERE/'source_comparison.json').write_text(json.dumps({'baseline_source':'official pinned commits; local edits are preserved and excluded','local_differences':differences},indent=2)+'\n')
    return files

def build():
    files=freeze()
    for name in ('baseline_runtime.py','author_worker.py','eval_worker.py','eval_protocol_v32.py','evaluation_source.json','notebook_session.py'):
        files[name]=(PARENT/name).read_bytes()
    for name in ('campaign_runtime.py','campaign_state.py','campaign_worker.py','campaign_session.py','retrieval_worker.py'):
        files[name]=(HERE/name).read_bytes()
    blob=io.BytesIO()
    with zipfile.ZipFile(blob,'w',zipfile.ZIP_DEFLATED) as z:
        for name,data in sorted(files.items()):
            info=zipfile.ZipInfo(name,(2026,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,data)
    bundle=blob.getvalue();(HERE/'campaign_code.zip').write_bytes(bundle)
    legacy=json.loads((PARENT/'official_baselines_n200.ipynb').read_text(encoding='utf8'))
    source=''.join(legacy['cells'][1]['source']);tree=ast.parse(source)
    cfg=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CONFIG' for t in n.targets))
    cfg.update(run_name='official_om_n200_e16_s300_v4',methods=['o_attack','m_attack_v2'],
      method_settings={'o_attack':{'seed':2023,'alpha':1.0,'batch_size':1,'overrides':{}},
                       'm_attack_v2':{'seed':2023,'alpha':0.005,'batch_size':10,'overrides':{}}},
      retrieval={'reference_root':'','coco_zip_url':'https://s3.amazonaws.com/images.cocodataset.org/zips/train2014.zip','coco_sha256':'ede4087e640bddba550e090eae701092534b554b42b05ac33f0300b984b31775','allow_download':True},
      verify_first_batch=True,session_hours=7.5,run_evaluation=True)
    cfg.pop('mpc',None);cfg.pop('saga',None);cfg.pop('alpha',None)
    cfg['model_revisions']['laion/CLIP-ViT-B-32-laion2B-s34B-b79K']='1a25a446712ba5ee05982a381eed697ef9b435cf'
    (HERE/'default_config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    config="""# Change this cell for another account or a new experiment.
from pathlib import Path
import os
import marimo as mo
CAMPAIGN_CONFIG = """+pprint.pformat(cfg,width=110,sort_dicts=False)+"""
CAMPAIGN_ROOT = Path('/kaggle/working/baselines' if Path('/kaggle').exists() else '/tmp/vlm_baselines' if Path('/marimo').exists() else './baselines')/CAMPAIGN_CONFIG['run_name']
"""
    boot="""# Locked official algorithms; dataset and credentials are not embedded.
import base64 as _cb, hashlib as _ch, io as _ci, zipfile as _cz
import sys as _cs, subprocess as _cp, shutil as _csh, importlib.util as _cu
for _module,_package in [('uv','uv'),('googleapiclient','google-api-python-client'),('pydata_google_auth','pydata-google-auth'),('dotenv','python-dotenv')]:
    if _cu.find_spec(_module) is None:
        _uv=_csh.which('uv')
        _cp.run([_uv,'pip','install','--python',_cs.executable,_package] if _uv else [_cs.executable,'-m','pip','install',_package],check=True)
_bundle=_cb.b64decode("""+repr(base64.b64encode(bundle).decode())+""")
assert _ch.sha256(_bundle).hexdigest()=="""+repr(hashlib.sha256(bundle).hexdigest())+"""
CAMPAIGN_ROOT.mkdir(parents=True,exist_ok=True)
with _cz.ZipFile(_ci.BytesIO(_bundle)) as _archive:
    for _item in _archive.infolist():
        _path=(CAMPAIGN_ROOT/_item.filename).resolve()
        if not _path.is_relative_to(CAMPAIGN_ROOT.resolve()): raise ValueError('Unsafe bundle path')
        _data=_archive.read(_item)
        if _path.exists() and _path.read_bytes()!=_data: raise ValueError('Source changed; use a new run_name')
        _path.parent.mkdir(parents=True,exist_ok=True)
        if not _path.exists(): _path.write_bytes(_data)
os.environ['PYTHONPATH']=str(CAMPAIGN_ROOT.resolve())+os.pathsep+os.environ.get('PYTHONPATH','')
if str(CAMPAIGN_ROOT.resolve()) not in _cs.path: _cs.path.insert(0,str(CAMPAIGN_ROOT.resolve()))
from campaign_session import CampaignSession as _CampaignSession
campaign_session=_CampaignSession(CAMPAIGN_ROOT,CAMPAIGN_CONFIG)
campaign_start=mo.ui.run_button(label='Chạy / tiếp tục O-Attack + M-Attack-V2')
campaign_refresh=mo.ui.refresh(options=['10s','30s','1m'],default_interval='30s')
mo.vstack([campaign_session.auth_ui(),campaign_start])
"""
    launch="if campaign_start.value:\n    campaign_session.launch()\n"
    monitor="campaign_refresh.value\nmo.vstack([campaign_refresh,campaign_session.monitor()])\n"
    (HERE/'molab_setup_cell.py').write_text(config+boot,encoding='utf8')
    cells=[config,boot,launch,monitor]
    native='import marimo\n__generated_with = "0.25.2"\napp=marimo.App()\n\n'
    signatures=[('', 'CAMPAIGN_CONFIG, CAMPAIGN_ROOT, mo, os'),('CAMPAIGN_CONFIG, CAMPAIGN_ROOT, mo, os','campaign_session, campaign_start, campaign_refresh'),('campaign_start, campaign_session',''),('campaign_refresh, campaign_session, mo','')]
    for code,(args,returns) in zip(cells,signatures):
        native+='@app.cell\ndef _('+args+'):\n'+''.join('    '+line+'\n' for line in code.rstrip().splitlines())+'    return '+returns+'\n\n'
    native+='if __name__=="__main__":\n    app.run()\n'
    ast.parse(native);(HERE/'official_om_molab.py').write_text(native,encoding='utf8')
    jupyter=[config.replace('import marimo as mo\n',''),boot[:boot.index('campaign_start=')]+"\nimport getpass\n_url=campaign_session.auth_url()\nif _url:\n    from IPython.display import display, Markdown\n    display(Markdown('[Connect Drive]('+_url+')'))\n    campaign_session.connect(getpass.getpass('OAuth callback URL: '))\nfor _key in ('OPENAI_API_KEY','GEMINI_API_KEY'):\n    if not os.environ.get(_key):\n        _value=getpass.getpass(_key+' (empty to defer): ')\n        if _value: campaign_session.set_secret(_key,_value)\n",'campaign_session.launch()\n','campaign_session.status()\n']
    nb={'cells':[{'id':f'om-cell-{i}','cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':c.splitlines(keepends=True)} for i,c in enumerate(jupyter)], 'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'nbformat':4,'nbformat_minor':5}
    (HERE/'official_om_kaggle.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf8')
    (HERE/'build_manifest.json').write_text(json.dumps({'bundle_sha256':hashlib.sha256(bundle).hexdigest(),'bundle_bytes':len(bundle),'author_files':sum(len(x['files']) for x in json.loads(files['author/SOURCE_LOCK.json']).values())},indent=2)+'\n')
    print('Built',len(bundle),'bytes; frozen sources verified against official commits.')

if __name__=='__main__':build()
