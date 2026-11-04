"""Provision ATA inputs through the author's unchanged COCO retrieval pipeline."""
import concurrent.futures, os, shutil, sys, zipfile
from pathlib import Path
from campaign_runtime import download_url, read_json, write_json, sha, extract, stage

def extract_coco_archive(archive,reference,expected=82783):
    """Resume extraction after a terminated session, publishing each JPEG atomically."""
    reference=Path(reference).resolve();reference.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        images=[i for i in z.infolist() if i.filename.startswith('train2014/') and i.filename.endswith('.jpg')]
        if len(images)!=expected:raise ValueError(f'Expected all {expected} COCO train2014 images')
        for info in images:
            path=(reference/info.filename).resolve()
            if not path.is_relative_to(reference):raise ValueError('Unsafe COCO ZIP path')
            if path.exists() and path.stat().st_size==info.file_size:continue
            path.parent.mkdir(parents=True,exist_ok=True)
            temp=path.with_suffix('.jpg.partial')
            with z.open(info) as src,temp.open('wb') as out:shutil.copyfileobj(src,out)
            temp.replace(path)
    write_json(reference/'extraction_complete.json',{'images':len(images),'archive_bytes':Path(archive).stat().st_size})

def ensure_retrieval(root,cfg,drive):
    rows=read_json(root/'selected_pairs.json');dest=root/'retrieval';dest.mkdir(exist_ok=True)
    remote=cfg['drive_prefix'].strip('/')+'/'+cfg['run_name']+'/assets/retrieval.zip'
    archive=root/'retrieval.zip'
    if drive and not archive.exists():
        folder,name=remote.rsplit('/',1);parent=drive.folder(folder,create=False)
        item=drive.find(parent,name) if parent else None
        if item:drive.download(item['id'],archive,item.get('md5Checksum'))
    if archive.exists():extract(archive,root)
    def complete(stem):return all((dest/stem/f'{i}.jpg').exists() for i in (1,2))
    assets=read_json(root/'released_retrieval_assets.json')
    release=read_json(root/'author/SOURCE_LOCK.json')['m_attack_v2']
    jobs=[]
    for row in rows:
        for rank in (1,2):
            rel=f'resources/retrieved_embeddings/{row["stem"]}/{rank}.jpg'
            if rel in assets:jobs.append((rel,dest/row['stem']/f'{rank}.jpg',assets[rel]))
    def fetch(job):
        rel,path,checksum=job
        url=release['repository'].replace('https://github.com/','https://raw.githubusercontent.com/')+'/'+release['commit']+'/'+rel
        return download_url(url,path,checksum)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:list(pool.map(fetch,jobs))
    missing=[row for row in rows if not complete(row['stem'])]
    if missing:
        stage(root,'preparing_retrieval',method='m_attack_v2',missing=len(missing),reference='COCO train2014')
        given=cfg['retrieval'].get('reference_root','')
        reference=Path(given).resolve() if given else root.parent/'datasets/coco2014'
        marker=reference/'extraction_complete.json'
        if not marker.exists() or read_json(marker).get('images')!=82783:
            if given and (reference/'train2014').exists():
                if len(list((reference/'train2014').glob('*.jpg')))!=82783:raise ValueError('External COCO reference is incomplete')
            else:
                if not cfg['retrieval']['allow_download']:raise FileNotFoundError('Set retrieval.reference_root to COCO train2014, or allow_download=True')
                reference.mkdir(parents=True,exist_ok=True)
                zip_path=reference.parent/'train2014.zip'
                download_url(cfg['retrieval']['coco_zip_url'],zip_path,cfg['retrieval'].get('coco_sha256'))
                extract_coco_archive(zip_path,reference)
        upstream=root/'author/m_attack_v2'
        link=upstream/'resources/images/coco';link.parent.mkdir(parents=True,exist_ok=True)
        if not link.exists():link.symlink_to(reference,target_is_directory=True)
        query=root/'retrieval_queries/images';query.mkdir(parents=True,exist_ok=True)
        for row in missing:
            path=root/'data'/row['target'];out=query/path.name
            if not out.exists():shutil.copyfile(path,out)
        import retrieval
        # Public author entry point controls embeddings, ranking and copying.
        previous=Path.cwd();os.chdir(upstream)
        try:retrieval.main(dataset_type='coco',target_images_dir=str(query.parent),device='cuda:0')
        finally:os.chdir(previous)
        for row in missing:
            for rank in (1,2):
                src=upstream/'resources/retrieved_embeddings'/row['stem']/f'{rank}.jpg'
                if not src.exists():raise FileNotFoundError('Author retrieval incomplete: '+str(src))
                out=dest/row['stem']/f'{rank}.jpg';out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,out)
    if not all(complete(row['stem']) for row in rows):raise ValueError('Incomplete ATA inputs; target replication is prohibited')
    manifest={row['stem']:{str(i):sha(dest/row['stem']/f'{i}.jpg') for i in (1,2)} for row in rows}
    write_json(root/'retrieval_manifest.json',{'inputs':manifest,'reference':'author released assets + official retrieval on COCO train2014', 'settings':cfg['retrieval']})
    if not archive.exists():
        temp=archive.with_suffix('.tmp')
        with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED) as z:
            for p in dest.rglob('*.jpg'):z.write(p,p.relative_to(root).as_posix())
            z.write(root/'retrieval_manifest.json','retrieval_manifest.json')
        temp.replace(archive)
    if drive:drive.put(archive,remote)
    print('ATA_INPUTS_VERIFIED',len(rows),'targets x 2 auxiliary images',flush=True)
