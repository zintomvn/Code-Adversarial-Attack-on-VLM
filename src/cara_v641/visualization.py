"""Migrated directly from CARA v6.4.1; see SOURCE_MAP.json and ATTRIBUTION.md."""
from .config import INSTALL_PACKAGES, RUN_ATTACK, RUN_EVALUATION, RUN_VISUALIZATION, NUM_SAMPLES, START_INDEX, SEED, DATA_ROOT, SOURCE_DIR, TARGET_DIR, PAIR_MANIFEST, OUTPUT_DIR, K, STEPS, XAI_METHOD, EPSILON, ALPHA, MOMENTUM, TEMPERATURE, OMEGA, CROP_AREA, ALPHA_GLOBAL, ALPHA_CROP, SAVE_BRANCH_IMAGES, SAVE_EVERY, SAVE_CROP_EVERY, ATTENTION_LAYER, NEGATIVE_RETRIES, CAPTION_ATTEMPTS, IMAGE_SIZE, CLIP_G14_ID, CLIP_B16_ID, CLIP_B32_ID, QWEN_ID, EVALUATION_ID, CLIP_DEVICES, ATTACK_DEVICE, QWEN_MAX_MEMORY_GIB, IMAGE_WEIGHT, TEXT_WEIGHT, CAPTION_TEMPERATURE, CAPTION_TOP_P, EVALUATION_PROMPT, RUN_ID, MODEL_REVISIONS
from .common import *
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


class CARAVisualization:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)

    def folder(self, sample_id):
        return self.output_dir / f"{sample_id:05d}"

    def steps(self, sample_id):
        folder = self.folder(sample_id) / "steps"
        files = sorted(folder.glob("step_*.png"))
        fig, axes = plt.subplots(2, math.ceil(len(files) / 2), figsize=(18, 7))
        for axis, path in zip(axes.ravel(), files):
            axis.imshow(Image.open(path))
            axis.set_title(path.stem)
            axis.axis("off")
        for axis in axes.ravel()[len(files):]:
            axis.axis("off")
        plt.tight_layout()
        plt.show()

    def attention(self, sample_id, role="adversarial"):
        files = sorted((self.folder(sample_id) / "xai").glob(f"{role}_*.npz"))
        if not files:
            print(f"No {role} XAI maps saved")
            return
        fig, axes = plt.subplots(2, math.ceil(len(files) / 2), figsize=(18, 6))
        for axis, path in zip(axes.ravel(), files):
            array = np.load(path)["aggregate"]
            axis.imshow(array, cmap="magma")
            axis.set_title(path.stem)
            axis.axis("off")
        for axis in axes.ravel()[len(files):]:
            axis.axis("off")
        plt.tight_layout()
        plt.show()

    def overlay(self, sample_id, role="source", step=0):
        folder = self.folder(sample_id)
        image_file = folder / f"{role}.png"
        if role == "adversarial":
            image_file = folder / "steps" / f"step_{step:03d}.png"
        heat_file = folder / "xai" / f"{role}_{step:03d}.npz"
        image = Image.open(image_file).convert("RGB")
        heat = torch.from_numpy(np.load(heat_file)["aggregate"])
        heat = F.interpolate(heat[None, None], size=(image.height, image.width),
                             mode="bilinear", align_corners=False)[0, 0].numpy()
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(image)
        axes[1].imshow(heat, cmap="magma")
        axes[2].imshow(image)
        axes[2].imshow(heat, cmap="magma", alpha=0.50)
        for ax, title in zip(axes, ["Image", "XAI heatmap", "Overlay"]):
            ax.set_title(title)
            ax.axis("off")
        plt.tight_layout()
        plt.show()

    def source_target_attention(self, sample_id):
        self.overlay(sample_id, "source", 0)
        self.overlay(sample_id, "target", 0)

    def similarity(self, sample_id):
        progress = read_json(self.folder(sample_id) / "progress.json")
        frame = pd.DataFrame(progress["history"])
        ax = frame.plot(x="step", y=["target_similarity", "max_negative_similarity"], figsize=(10, 4))
        ax.set_ylabel("Cosine similarity")
        plt.show()
        ax = frame.plot(x="step", y=["global_loss", "crop_loss", "loss"], figsize=(10, 4))
        ax.set_ylabel("CE: global / crop / diagnostic sum")
        plt.show()

    def dispersion(self, sample_id):
        files = sorted((self.folder(sample_id) / "xai").glob("adversarial_*.json"))
        rows = []
        for path in files:
            record = read_json(path)
            if record.get("dispersion"):
                rows.append({"step": record["step"], **record["dispersion"]})
        if rows:
            pd.DataFrame(rows).plot(x="step", y=["entropy", "effective_support", "variance"],
                                    figsize=(10, 4), title="Attention dispersion")
            plt.show()

    def caption_space(self, sample_id):
        path = self.folder(sample_id) / "caption_embeddings.npz"
        bank = read_json(self.folder(sample_id) / "bank.json")
        groups = ["source_captions", "target_captions", "negative_captions"]
        data = np.load(path)
        vectors = np.concatenate([data[name] for name in groups], axis=0)
        points = PCA(n_components=2).fit_transform(vectors)
        for group_index, name in enumerate(groups):
            subset = points[group_index * K:(group_index + 1) * K]
            plt.scatter(subset[:, 0], subset[:, 1], label=name)
            for i, (x, y) in enumerate(subset):
                plt.annotate(str(i + 1), (x, y), fontsize=8)
        plt.legend()
        plt.title("CLIP caption embeddings — PCA")
        plt.show()


    def _crop_frame(self, sample_id, step):
        folder=self.folder(sample_id)
        trace_path=folder/'crop_trace'/f'step_{step:03d}.json'
        if not trace_path.exists():raise FileNotFoundError(f'No actual pre-update crop snapshot: {trace_path}')
        record=read_json(trace_path)
        image=Image.open(folder/record['input_image']).convert('RGB')
        with np.load(folder/record['attention_npz']) as data:heat=torch.from_numpy(data['aggregate'].copy())
        full=F.interpolate(heat[None,None].float(),(image.height,image.width),mode='bilinear',align_corners=False)[0,0]
        full=full/full.sum()
        return record,image,full.numpy()

    def crop_attention(self, sample_id, steps=(1,30,150,300), save=True):
        """Exact BEFORE-update images and the valid map that actually chose each box."""
        from matplotlib.patches import Rectangle
        frames=[self._crop_frame(sample_id,step) for step in steps]
        if not frames:raise ValueError('Choose at least one saved crop step')
        vmax=max(float(p.max()) for _,_,p in frames)
        fig,axes=plt.subplots(len(frames),5,figsize=(19,3.7*len(frames)),squeeze=False,layout='constrained')
        rows=[]
        for row,(record,image,probability) in enumerate(frames):
            top,left,height,width=record['crop'];a=axes[row]
            reused=' [REUSED]' if record['attention_reused'] else ''
            a[0].imshow(image)
            a[0].add_patch(Rectangle((left,top),width,height,fill=False,edgecolor='cyan',linewidth=2))
            a[0].set_title(f"Before update {record['step']} | guide {record['attention_role']}:{record['attention_step']}{reused}",fontsize=9)
            a[0].text(0,-.05,record.get('guide_generated_caption',''),transform=a[0].transAxes,fontsize=8,wrap=True,va='top')
            if record['attention_reused']:
                a[0].text(0,-.22,'Latest: '+record.get('latest_generated_caption',''),transform=a[0].transAxes,fontsize=8,wrap=True,va='top',color='darkred')
            heat=a[1].imshow(probability,cmap='magma',vmin=0,vmax=vmax)
            a[1].add_patch(Rectangle((left,top),width,height,fill=False,edgecolor='cyan',linewidth=2))
            a[1].set_title(f"Guide mass in box: {record['attention_mass_in_crop']:.3f}; H={record['dispersion']['entropy']:.3f}",fontsize=9)
            fig.colorbar(heat,ax=a[1],shrink=.8,label='Normalized image-grid probability')
            crop_image=Image.open(self.folder(sample_id)/record['crop_image']).convert('RGB')
            a[2].imshow(crop_image);a[2].set_title(f'Actual input crop {width}x{height}')
            a[3].imshow(probability[top:top+height,left:left+width],cmap='magma',vmin=0,vmax=vmax)
            a[3].set_title('Attention in crop (same color scale)')
            a[4].plot(np.linspace(0,1,image.width),probability.sum(0),label='Column marginal')
            a[4].plot(np.linspace(0,1,image.height),probability.sum(1),label='Row marginal')
            a[4].set(xlabel='Normalized position',ylabel='Probability mass',title='Spatial distribution')
            a[4].legend(fontsize=8);a[4].grid(alpha=.2)
            for axis in a[:4]:axis.axis('off')
            rows.append({k:record[k] for k in ('step','attention_role','attention_step','attention_reused',
                'attention_mass_in_crop','actual_crop_area','mass_vs_uniform')})
        fig.suptitle(f'Sample {sample_id}: actual crop inputs and guide attention (not post-update maps)')
        table=pd.DataFrame(rows)
        if save:
            dest=self.folder(sample_id)/'visualizations';dest.mkdir(exist_ok=True)
            stem=f'crop_attention_{min(steps):03d}_{max(steps):03d}'
            fig.savefig(dest/(stem+'.png'),dpi=130,bbox_inches='tight')
            table.to_csv(dest/(stem+'.csv'),index=False)
        plt.show()
        return table

    def concept_attention(self, sample_id, update_step=1, native=False, save=True):
        """Per-concept maps of the actual guide, with matched generated-token labels."""
        record,_,_=self._crop_frame(sample_id,update_step)
        folder=self.folder(sample_id);meta=read_json(folder/record['attention_json'])
        labels=meta.get('concept_labels',[]);prefix='native_concept_' if native else 'concept_'
        with np.load(folder/record['attention_npz']) as data:
            maps=[(label,data[f'{prefix}{i}'].copy()) for i,label in enumerate(labels) if f'{prefix}{i}' in data]
        if not maps:raise ValueError(f'No saved per-concept maps for {record["attention_npz"]}, native={native}')
        cols=min(3,len(maps));rows=math.ceil(len(maps)/cols)
        fig,axes=plt.subplots(rows,cols,figsize=(5*cols,4*rows),squeeze=False,layout='constrained')
        vmax=max(float(value.max()) for _,value in maps)
        info=meta.get('metadata',{});tokens=info.get('generated_tokens',[])
        for axis,(label,value) in zip(axes.ravel(),maps):
            artist=axis.imshow(value,cmap='magma',vmin=0,vmax=vmax)
            positions=info.get('concept_token_offsets',{}).get(label,[])
            matched=''.join(tokens[i] for i in positions if i<len(tokens))
            entropy=XAIMap.dispersion(torch.from_numpy(value))['entropy']
            axis.set_title(f'{label} | H={entropy:.3f}\nGenerated tokens: {matched}',fontsize=10)
            axis.axis('off');fig.colorbar(artist,ax=axis,shrink=.8)
        for axis in axes.ravel()[len(maps):]:axis.axis('off')
        grid='native Qwen vision grid' if native else 'normalized32x32 grid'
        reused=' [REUSED guide]' if record['attention_reused'] else ''
        fig.suptitle(f"Sample {sample_id}, crop update {update_step}; map {record['attention_step']}{reused}; {grid}")
        if save:
            dest=folder/'visualizations';dest.mkdir(exist_ok=True)
            suffix='_native' if native else ''
            fig.savefig(dest/f'concept_attention_update_{update_step:03d}{suffix}.png',dpi=130,bbox_inches='tight')
        plt.show()
        return fig

    def crop_attention_distribution(self, sample_id, save=True):
        folder=self.folder(sample_id)
        records=[read_json(p) for p in sorted((folder/'crop_trace').glob('step_*.json'))]
        if not records:raise FileNotFoundError('No crop_trace snapshots for this sample')
        table=pd.DataFrame([{'step':r['step'],'attention_step':r['attention_step'],
            'attention_reused':r['attention_reused'],'attention_mass_in_crop':r['attention_mass_in_crop'],
            'actual_crop_area':r['actual_crop_area'],'mass_vs_uniform':r['mass_vs_uniform'],
            **r['dispersion']} for r in records])
        fig,axes=plt.subplots(1,3,figsize=(17,4.5),layout='constrained')
        axes[0].plot(table['step'],table['attention_mass_in_crop'],'o-',label='Attention mass in actual crop')
        axes[0].plot(table['step'],table['actual_crop_area'],'--',label='Uniform-map mass (= area)')
        reused=table[table['attention_reused']]
        axes[0].scatter(reused['step'],reused['attention_mass_in_crop'],marker='x',color='red',label='Reused guide')
        axes[0].set(xlabel='Update',ylabel='Probability mass',ylim=(0,1),title='Attention coverage by crop')
        axes[1].plot(table['step'],table['entropy'],'o-',label='Normalized entropy')
        axes[1].plot(table['step'],table['effective_support'],'o-',label='Effective support')
        axes[1].set(xlabel='Update',ylabel='Normalized value',ylim=(0,1.05),title='Guide dispersion (reused != fresh)')
        for record in (records[0],records[-1]):
            _,_,probability=self._crop_frame(sample_id,record['step'])
            sorted_mass=np.sort(probability.ravel())[::-1]
            axes[2].plot(np.arange(1,len(sorted_mass)+1)/len(sorted_mass),sorted_mass.cumsum(),
                         label=f"Update {record['step']}, guide {record['attention_step']}")
        axes[2].plot([0,1],[0,1],'--',color='gray',label='Uniform')
        axes[2].set(xlabel='Fraction of highest-attention pixels',ylabel='Cumulative attention mass',
                    title='Spatial attention concentration',xlim=(0,1),ylim=(0,1.01))
        for axis in axes:axis.legend(fontsize=8);axis.grid(alpha=.2)
        fig.suptitle(f'Sample {sample_id}: crop guide probability distributions')
        if save:
            dest=folder/'visualizations';dest.mkdir(exist_ok=True)
            fig.savefig(dest/'crop_attention_distribution.png',dpi=130,bbox_inches='tight')
            table.to_csv(dest/'crop_attention_distribution.csv',index=False)
        plt.show()
        return table
