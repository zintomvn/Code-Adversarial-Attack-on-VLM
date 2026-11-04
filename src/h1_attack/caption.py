"""No-API local HF caption generation. No target text or judging."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image

PROMPT='Describe this image in one concise sentence, no longer than 20 words.'
MODELS={'qwen':'Qwen/Qwen2.5-VL-3B-Instruct','llava':'llava-hf/llava-1.5-7b-hf'}


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pixel_digest(image):
    a=np.array(image.convert('RGB'))
    return hashlib.sha256(str(a.shape).encode()+a.tobytes()).hexdigest()


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)


def read_verified(path,sha256):
    if digest(path)!=sha256:raise ValueError(f'image hash mismatch: {path}')
    with Image.open(path) as image:return image.convert('RGB').copy()


def epsilon_check(source,adv,epsilon):
    with Image.open(source) as im:s=np.array(im.convert('RGB')).astype(np.int16)
    with Image.open(adv) as im:a=np.array(im.convert('RGB')).astype(np.int16)
    if s.shape!=a.shape:raise ValueError('source/adversarial shape mismatch')
    linf=int(np.abs(a-s).max())
    if linf>epsilon:raise ValueError(f'saved epsilon violation: {linf}>{epsilon}')
    return linf


def cache_key(pixel_hash,protocol):
    return hashlib.sha256(json.dumps({'pixel':pixel_hash,'protocol':protocol},sort_keys=True).encode()).hexdigest()


def caption_flags(text,tokens,max_tokens,ended_eos):
    return {'word_count':len(text.split()),'over_word_limit':len(text.split())>20,
            'truncated':tokens>=max_tokens and not ended_eos,'ended_eos':bool(ended_eos)}


def evaluate_rows(rows,protocol,output,generate):
    output=Path(output);cache=output/'cache';cache.mkdir(parents=True,exist_ok=True)
    results=[]
    for row in rows:
        key=cache_key(row['pixel_sha256'],protocol);path=cache/f'{key}.json'
        cached=False
        if path.exists():
            payload=json.loads(path.read_text(encoding='utf-8'))
            if payload.get('cache_key')!=key:raise ValueError('corrupt caption cache identity')
            cached=payload.get('status')=='captioned'
        if not cached:
            started=time.perf_counter()
            try:
                payload={'status':'captioned',**generate(row['path'])}
            except Exception as exc:
                payload={'status':'error','error_type':type(exc).__name__,'error':str(exc)[:500]}
            payload.update({'cache_key':key,'generation_seconds':time.perf_counter()-started})
            atomic_json(path,payload)
        result={**row,**payload,'cache_hit':cached};results.append(result)
        atomic_json(output/'progress.json',{'protocol':protocol,'rows':results,
                    'completed_rows':len(results),'total_rows':len(rows),'victim_asr':None,'judge':'not run'})
        print(json.dumps({'id':row['id'],'status':payload['status'],'cache_hit':cached}),flush=True)
    atomic_json(output/'captions.json',{'protocol':protocol,'rows':results,'victim_asr':None,'judge':'not run'})
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',choices=MODELS,default='qwen')
    p.add_argument('--revision',default='main',help='resolved to immutable commit before loading')
    p.add_argument('--local-only',action='store_true')
    p.add_argument('--dtype',choices=['float16','bfloat16','float32'],default='float16')
    p.add_argument('--device',default='cuda')
    a=p.parse_args();manifest=json.loads(a.manifest.read_text(encoding='utf-8'));root=a.manifest.parent.resolve()
    rows=manifest['rows']
    for row in rows:
        path=(root/row['path']).resolve()
        if not path.is_relative_to(root):raise ValueError('manifest path escapes bundle')
        image=read_verified(path,row['file_sha256'])
        if pixel_digest(image)!=row['pixel_sha256']:raise ValueError('pixel identity mismatch')
        if row['role']=='adversarial':epsilon_check(root/row['source_path'],path,16)
    import torch
    import transformers
    from huggingface_hub import HfApi,snapshot_download
    from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration,LlavaForConditionalGeneration
    model_id=MODELS[a.model]
    if a.local_only:
        snapshot=Path(snapshot_download(model_id,revision=a.revision,local_files_only=True))
        revision=snapshot.name
    else:
        revision=HfApi().model_info(model_id,revision=a.revision).sha
        snapshot=Path(snapshot_download(model_id,revision=revision))
    if len(revision)!=40:raise ValueError('model revision must resolve to a commit SHA')
    cls=Qwen2_5_VLForConditionalGeneration if a.model=='qwen' else LlavaForConditionalGeneration
    processor=AutoProcessor.from_pretrained(str(snapshot),local_files_only=True)
    model=cls.from_pretrained(str(snapshot),torch_dtype=getattr(torch,a.dtype),
                             attn_implementation='eager',local_files_only=True).to(a.device).eval()
    eos=model.generation_config.eos_token_id
    eos_ids=eos if isinstance(eos,list) else [eos]
    protocol={'schema':'caption-local-v1','model_id':model_id,'resolved_revision':revision,
              'processor_revision':revision,'processor_class':type(processor).__name__,
              'image_processor_config':processor.image_processor.to_dict(),
              'tokenizer_class':type(processor.tokenizer).__name__,
              'prompt':PROMPT,'do_sample':False,'num_beams':1,
              'max_new_tokens':64,'batch_size':1,'dtype':a.dtype,'device':a.device,
              'attention_implementation':'eager','torch':torch.__version__,'transformers':transformers.__version__,
              'snapshot_json_sha256':{f.name:digest(f) for f in snapshot.glob('*.json')},
              'generation_config':model.generation_config.to_dict(),
              'baseline_comparability':'diagnostic Qwen2.5-VL; NOT baseline Qwen3' if a.model=='qwen' else 'exact baseline model/settings not established',
              'evaluator_code_sha256':digest(__file__)}
    # Config must be JSON serializable before generation/cache identity is used.
    json.dumps(protocol)
    atomic_json(a.output/'run_identity.json',{'protocol':protocol,'manifest_sha256':digest(a.manifest),
                                            'manifest_provenance':manifest.get('provenance')})
    @torch.inference_mode()
    def generate(relative):
        image=Image.open(root/relative).convert('RGB')
        if a.model=='qwen':
            messages=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':PROMPT}]}]
            text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        else:
            text=f'USER: <image>\n{PROMPT} ASSISTANT:'
        inputs=processor(text=[text],images=[image],return_tensors='pt').to(a.device)
        # Integer token IDs remain integer; floating image inputs match model dtype.
        for name,value in inputs.items():
            if torch.is_floating_point(value):inputs[name]=value.to(getattr(torch,a.dtype))
        output=model.generate(**inputs,max_new_tokens=64,do_sample=False,num_beams=1,use_cache=True)
        generated=output[0,inputs['input_ids'].shape[1]:]
        tokens=generated.tolist();caption=processor.decode(generated,skip_special_tokens=True).strip()
        return {'caption':caption,'generated_token_ids':tokens,'new_tokens':len(tokens),
                **caption_flags(caption,len(tokens),64,bool(tokens and tokens[-1] in eos_ids))}
    results=evaluate_rows(rows,protocol,a.output,generate)
    if any(r['status']=='error' for r in results):raise SystemExit(2)


if __name__=='__main__':main()
