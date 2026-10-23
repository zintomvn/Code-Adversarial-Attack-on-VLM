"""One editable configuration, private credentials and visible campaign progress."""
import json, os, subprocess, sys
from pathlib import Path
from notebook_session import Session
from campaign_runtime import read_json, write_json, credentials, validate, verify_sources

class CampaignSession(Session):
    def launch(self):
        validate(self.config);verify_sources(self.root)
        if self.config['backup_drive']:credentials()
        pointer=self.root/'controller_pid.json'
        if pointer.exists():
            prior=read_json(pointer)
            try:
                stat=Path(f'/proc/{prior["pid"]}/stat').read_text();fields=stat[stat.rfind(')')+2:].split()
                if fields[0]!='Z' and fields[19]==prior['start_ticks']:
                    print('Campaign already running:',prior['pid']);return prior['pid']
            except OSError:pass
        previous=self.root/'config.json'
        if previous.exists():
            old=read_json(previous)
            for key in ('n','epsilon','steps','methods','method_settings','data_sha256','model_revisions','retrieval'):
                if old[key]!=self.config[key]:raise ValueError('Attack config changed; choose a new run_name: '+key)
        if subprocess.run(['nvidia-smi'],capture_output=True).returncode:raise RuntimeError('Select an NVIDIA GPU first')
        write_json(previous,self.config);(self.root/'logs').mkdir(exist_ok=True)
        env=os.environ.copy();env['PYTHONPATH']=str(self.root.resolve())+os.pathsep+env.get('PYTHONPATH','')
        # Keep the old campaign's optional Python startup hook out of this run.
        env.pop('BASELINE_DRIVE_TRANSPORT_RETRY',None)
        with (self.root/'logs/controller.log').open('a') as log:
            child=subprocess.Popen([sys.executable,'-u',str(self.root/'campaign_runtime.py'),str(self.root)],env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        raw=Path(f'/proc/{child.pid}/stat').read_text();fields=raw[raw.rfind(')')+2:].split()
        write_json(pointer,{'pid':child.pid,'start_ticks':fields[19]})
        print('Campaign started/resumed:',child.pid);return child.pid

    def monitor(self):
        import marimo as mo
        data=self.status();s=data['stage'];panels=[mo.md('## Baseline campaign\n**'+s.get('stage','ready')+'** · '+s.get('method',s.get('model',''))),mo.ui.table(data['attacks'],selection=None)]
        evaluation=[]
        for folder in sorted((self.root/'evaluation').glob('*')):
            if not folder.is_dir():continue
            row={'model':folder.name,'captions_ok':0,'scores_ok':0,'errors':0}
            for kind in ('captions','scores'):
                p=folder/(kind+'.json')
                if p.exists():
                    records=read_json(p);row[kind+'_ok']=sum(v.get('status')=='ok' for v in records.values());row['errors']+=sum(v.get('status')=='error' for v in records.values())
            evaluation.append(row)
        if evaluation:panels+=[mo.md('### VLM / GPT scores'),mo.ui.table(evaluation,selection=None)]
        gpu=subprocess.run(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,memory.total','--format=csv,noheader'],capture_output=True,text=True)
        panels.append(mo.plain_text(gpu.stdout.strip()))
        for log in sorted((self.root/'logs').glob('*.log')):
            with log.open('rb') as stream:
                stream.seek(max(0,log.stat().st_size-2500));tail=stream.read().decode('utf8',errors='replace')
            lines=[x for x in tail.replace('\r','\n').splitlines() if x.strip()]
            panels += [mo.md('### '+log.name),mo.plain_text('\n'.join(lines[-16:]))]
        panels.append(mo.md('[Drive checkpoints](https://drive.google.com/drive/folders/'+self.config['drive_parent_id']+') · giữ cùng run_name để chạy tiếp.'))
        return mo.vstack(panels)
