"""I/O adapters around immutable official entry points; never replace attack math."""
from __future__ import annotations
import argparse, importlib, json, os, sys
from pathlib import Path
from types import SimpleNamespace
from campaign_runtime import REGISTRY, install_storage, verify_sources, write_json, read_json, stage
from campaign_state import State, BatchLimitReached

def run_o_attack(root,state):
    import torch
    import oattack.run as run
    settings=state.cfg['method_settings']['o_attack']
    changes={
      'seed':settings['seed'],'data.clean_root':str(root/'inputs/source'),
      'data.target_root':str(root/'inputs/target'),'data.num_samples':state.cfg['n'],
      'data.batch_size':settings['batch_size'],'output.root':str(root/'author_output/o_attack'),
      'model.device':'cuda:0','optim.steps':state.cfg['steps'],'optim.epsilon':state.cfg['epsilon'],'optim.alpha':settings['alpha'],
      'text_objective.source_caption_json':str(root/'author/o_attack/data/text/bigscale_1000/nips17/caption.json'),
      'text_objective.target_caption_json':str(root/'author/o_attack/data/text/target_images_1000/1/caption.json'),**settings['overrides']}
    if settings['batch_size']!=1:raise ValueError('O-Attack released G3 uses batch_size=1; choose a separate adapter to change batching')
    run.parse_args=lambda:SimpleNamespace(config=str(root/'author/o_attack/configs/attack/default.yaml'),override=[k+'='+json.dumps(v) for k,v in changes.items()])
    original_load=run.load_config
    def load(*args):
        cfg=original_load(*args)
        from omegaconf import OmegaConf
        write_json(root/'o_attack_resolved.json',OmegaConf.to_container(cfg,resolve=True));return cfg
    run.load_config=load
    base_loader=torch.utils.data.DataLoader
    class PendingLoader:
        def __init__(self,*a,**kw):self.loader=base_loader(*a,**kw)
        def __iter__(self):
            for batch in self.loader:
                if all(Path(p).stem in state.done for p in batch[2]):continue
                yield batch
        def __len__(self):return len(self.loader)
    torch.utils.data.DataLoader=PendingLoader
    original_get=run.get_attack;last={}
    def get(name):
        original=original_get(name)
        def attack(**kw):
            state.before();last['source']=kw['image_org'].detach()/255;last['target']=kw['image_tgt'].detach()/255
            return original(**kw)
        return attack
    run.get_attack=get
    original_save=run.save_adversarial_batch
    def save(cfg,run_hash,adv,paths):
        outputs=original_save(cfg,run_hash,adv,paths)
        state.commit([(Path(p).stem,last['source'][i],last['target'][i],outputs[i]) for i,p in enumerate(paths)])
        return outputs
    run.save_adversarial_batch=save
    try:run.main()
    finally:torch.utils.data.DataLoader=base_loader

def run_m_attack(root,state):
    import torch, torchvision
    from torchvision import transforms
    from omegaconf import OmegaConf
    import generate_ad_sample_parallel as run
    from retrieval_worker import ensure_retrieval
    ensure_retrieval(root,state.cfg,state.drive)
    stage(root,'attacking',method='m_attack_v2',verification_batch=state.max_batches is not None)
    settings=state.cfg['method_settings']['m_attack_v2']
    cfg=OmegaConf.load(root/'author/m_attack_v2/config/ensemble_3models.yaml')
    changes={'data.cle_data_path':str(root/'inputs/source'),'data.tgt_data_path':str(root/'inputs/target'),
      'data.output':str(root/'author_output/m_attack_v2'),'data.num_samples':state.cfg['n'],
      'data.batch_size':settings['batch_size'],'data.retrieval_path':str(root/'retrieval'),
      'model.device':'cuda:0','model.device_ids':[0],'optim.steps':state.cfg['steps'],
      'optim.epsilon':state.cfg['epsilon'],'optim.alpha':settings['alpha'],**settings['overrides']}
    cfg=OmegaConf.merge(cfg,OmegaConf.from_dotlist([k+'='+json.dumps(v) for k,v in changes.items()]))
    if not cfg.optim.use_retrieval or cfg.model.target_num!=3:raise ValueError('This adapter requires released ATA with two auxiliary targets')
    write_json(root/'m_attack_v2_resolved.json',OmegaConf.to_container(cfg,resolve=True))
    transform=transforms.Compose([transforms.Resize(cfg.model.input_res,interpolation=transforms.InterpolationMode.BICUBIC),transforms.CenterCrop(cfg.model.input_res),transforms.ToTensor()])
    clean=run.ImageFolderWithPaths(cfg.data.cle_data_path,transform=transform)
    target=run.ImageFolderWithPaths(cfg.data.tgt_data_path,transform=transform)
    if [Path(p).stem for p,_ in clean.samples]!=[r['stem'] for r in state.rows]:raise ValueError('ImageFolder order differs from locked cohort')
    batch_size=cfg.data.batch_size
    pending=[]
    for batch in range((len(clean)+batch_size-1)//batch_size):
        ids=[Path(p).stem for p,_ in clean.samples[batch*batch_size:(batch+1)*batch_size]]
        if any(s in state.done for s in ids) and not all(s in state.done for s in ids):raise ValueError('Partial completed author batch')
        if not all(s in state.done for s in ids):pending.append(batch)
    class ResumeLoader:
        def __init__(self,loader,source=False):self.loader=loader;self.calls=0;self.source=source
        def __iter__(self):
            self.calls+=1
            # process_batch makes initial iterators, then recreates them for every batch.
            # Restore before the first pending batch's iterator RNG draw, not after it.
            if self.source and self.calls==2:state.restore()
            return iter(self.loader)
        def __len__(self):return len(self.loader)
    a=ResumeLoader(torch.utils.data.DataLoader(clean,batch_size=batch_size,shuffle=False,num_workers=4),True)
    b=ResumeLoader(torch.utils.data.DataLoader(target,batch_size=batch_size,shuffle=False,num_workers=4))
    original=run.attack_imgpair
    def attack(**kw):
        state.before(restore=False);original(**kw)
        items=[]
        for i,p in enumerate(kw['path_org']):
            stem=Path(p).stem;output=Path(cfg.data.output)/'img'/kw['config_hash']/Path(p).parent.name/(stem+'.png')
            if not output.exists():raise FileNotFoundError('Author output not saved: '+str(output))
            items.append((stem,kw['image_org'][i],kw['image_tgts'][i*cfg.model.target_num],output))
        state.commit(items)
    run.attack_imgpair=attack
    run.set_environment(settings['seed'])
    run.process_batch(0,0,cfg,pending,a,b,cfg.model.target_num,True,cfg.data.retrieval_path,run.hash_training_config(cfg),transform)

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('method');p.add_argument('--max-new-batches',type=int);p.add_argument('--model')
    args=p.parse_args();root=args.root.resolve();install_storage();verify_sources(root)
    if args.method=='evaluation':
        from eval_worker import evaluate
        return evaluate(root,args.model)
    spec=REGISTRY[args.method];author=root/'author'/args.method
    os.chdir(author);sys.path.insert(0,str(author))
    import torch
    if not torch.cuda.is_available():raise RuntimeError('GPU is required')
    from author_worker import pin_model_downloads
    if args.method=='m_attack_v2':pin_model_downloads(read_json(root/'config.json')['model_revisions'])
    state=State(root,args.method,args.max_new_batches)
    print('HARDWARE',json.dumps(state.environment),flush=True)
    try:globals()[spec['adapter']](root,state)
    except BatchLimitReached:print('VERIFIED_AUTHOR_BATCH',args.method,len(state.done),flush=True);return 0
    if len(state.done)!=state.cfg['n'] and args.max_new_batches is None:raise RuntimeError('Attack returned before cohort completion')
    state.manifest();return 0

if __name__=='__main__':sys.exit(main())
