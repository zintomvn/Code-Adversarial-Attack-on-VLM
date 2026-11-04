"""Migrated directly from CARA v6.4.1; see SOURCE_MAP.json and ATTRIBUTION.md."""
from .config import INSTALL_PACKAGES, RUN_ATTACK, RUN_EVALUATION, RUN_VISUALIZATION, NUM_SAMPLES, START_INDEX, SEED, DATA_ROOT, SOURCE_DIR, TARGET_DIR, PAIR_MANIFEST, OUTPUT_DIR, K, STEPS, XAI_METHOD, EPSILON, ALPHA, MOMENTUM, TEMPERATURE, OMEGA, CROP_AREA, ALPHA_GLOBAL, ALPHA_CROP, SAVE_BRANCH_IMAGES, SAVE_EVERY, SAVE_CROP_EVERY, ATTENTION_LAYER, NEGATIVE_RETRIES, CAPTION_ATTEMPTS, IMAGE_SIZE, CLIP_G14_ID, CLIP_B16_ID, CLIP_B32_ID, QWEN_ID, EVALUATION_ID, CLIP_DEVICES, ATTACK_DEVICE, QWEN_MAX_MEMORY_GIB, IMAGE_WEIGHT, TEXT_WEIGHT, CAPTION_TEMPERATURE, CAPTION_TOP_P, EVALUATION_PROMPT, RUN_ID, MODEL_REVISIONS
from .common import *
from .losses import CropGlobalLoss
from .regions import RegionSelector, XAIMap
class CARAAttack:
    def __init__(self, encoder, explainer, output_dir):
        self.encoder = encoder
        self.explainer = explainer
        self.loss_function = CropGlobalLoss(TEMPERATURE, OMEGA)
        self.selector = RegionSelector(IMAGE_SIZE, IMAGE_SIZE)
        self.output_dir = Path(output_dir)

    @torch.no_grad()
    def overwrite_updates(self, source, delta, global_momentum, crop_momentum, box,
                          global_gradient, crop_gradient):
        """Two proposals from delta_t; crop wins100% inside box, global wins outside."""
        top, left, height, width = box
        region = (slice(None), slice(None), slice(top,top+height), slice(left,left+width))
        if global_gradient.shape!=delta.shape or crop_gradient.shape!=delta[region].shape:
            raise ValueError('Gradient/proposal shape mismatch')
        new_global_momentum = MOMENTUM*global_momentum + global_gradient
        new_crop_momentum = crop_momentum.clone()
        new_crop_momentum[region] = MOMENTUM*crop_momentum[region] + crop_gradient
        def project(proposal, original):
            proposal=proposal.clamp(-EPSILON,EPSILON)
            return torch.maximum(torch.minimum(proposal,255-original),-original)
        global_proposal=project(delta+ALPHA_GLOBAL*new_global_momentum.sign(),source)
        crop_proposal=project(delta[region]+ALPHA_CROP*new_crop_momentum[region].sign(),source[region])
        merged=global_proposal.clone()
        merged[region]=crop_proposal  # REPLACE; never add global and crop updates.
        self.last_proposals={'global':global_proposal, 'crop':crop_proposal, 'box':list(box)}
        return merged.detach(), new_global_momentum.detach(), new_crop_momentum.detach()

    def save_xai(self, image, concepts, sample_dir, role, step, previous=None):
        xai_dir = sample_dir / "xai"
        xai_dir.mkdir(parents=True, exist_ok=True)
        info = {"step": step, "role": role, "method": XAI_METHOD, "valid": False}
        try:
            maps, metadata = self.explainer.maps(tensor_to_image(image), concepts)
            heatmap, per_concept = XAIMap.normalize(maps)
            info.update(valid=True, dispersion=XAIMap.dispersion(heatmap), metadata=metadata)
            data = {"aggregate": heatmap.numpy()}
            for index, (concept, value) in enumerate(per_concept.items()):
                data[f"concept_{index}"] = value.numpy()
                native=torch.as_tensor(maps[concept]).detach().float().cpu()
                data[f"native_concept_{index}"] = (native/native.sum()).numpy()
            np.savez_compressed(xai_dir / f"{role}_{step:03d}.npz", **data)
            info["concept_labels"] = list(per_concept)
        except Exception as error:
            info["error"] = f"{type(error).__name__}: {error}"
            info["metadata"] = getattr(error,"details",{})
            heatmap = previous
            info["fallback"] = heatmap is not None
            if heatmap is not None:
                np.savez_compressed(xai_dir / f"{role}_{step:03d}.npz", aggregate=heatmap.numpy())
        write_json(xai_dir / f"{role}_{step:03d}.json", info)
        return heatmap, info

    @torch.no_grad()
    def snapshot_crop(self, image, heat, box, sample_dir, step,
                      attention_role, attention_step, latest_refresh_step):
        """Persist the actual PRE-update view/box and the map which selected it."""
        sample_dir=Path(sample_dir)
        for name in ('crop_inputs','crops','crop_trace'):(sample_dir/name).mkdir(parents=True,exist_ok=True)
        top,left,height,width=box
        input_path=f'crop_inputs/step_{step:03d}.png'
        crop_path=f'crops/step_{step:03d}.png'
        picture=tensor_to_image(image)
        picture.save(sample_dir/input_path)
        picture.crop((left,top,left+width,top+height)).save(sample_dir/crop_path)
        map_stem=f'xai/{attention_role}_{attention_step:03d}'
        metadata=read_json(sample_dir/(map_stem+'.json'))
        latest_path=sample_dir/f'xai/adversarial_{latest_refresh_step:03d}.json'
        latest=read_json(latest_path) if latest_path.exists() else metadata
        distribution=F.interpolate(heat[None,None].float(),image.shape[-2:],mode='bilinear',align_corners=False)[0,0]
        distribution=distribution/distribution.sum()
        crop_mass=float(distribution[top:top+height,left:left+width].sum())
        actual_area=height*width/(image.shape[-2]*image.shape[-1])
        record={'step':step,'crop':list(box),'input_image':input_path,'crop_image':crop_path,
                'attention_role':attention_role,'attention_step':attention_step,
                'latest_refresh_step':latest_refresh_step,
                'attention_reused':latest_refresh_step!=attention_step,
                'attention_npz':map_stem+'.npz','attention_json':map_stem+'.json',
                'attention_mass_in_crop':crop_mass,'actual_crop_area':actual_area,
                'mass_vs_uniform':crop_mass/actual_area,
                'dispersion':XAIMap.dispersion(heat),
                'guide_generated_caption':metadata.get('metadata',{}).get('generated_caption',''),
                'latest_generated_caption':latest.get('metadata',{}).get('generated_caption',''),
                'latest_refresh_error':latest.get('error'),
                'matched_concepts':metadata.get('metadata',{}).get('matched_concepts',[]),
                'map_description':'normalized concept-mean attention, interpolated to image for crop mass'}
        write_json(sample_dir/f'crop_trace/step_{step:03d}.json',record)
        return record

    def attack(self, source, target, caption_bank, sample_id):
        sample_dir = self.output_dir / f"{sample_id:05d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        source = source.to(ATTACK_DEVICE)
        target = target.to(ATTACK_DEVICE)

        positive, negatives, selected = self.encoder.build_targets(source, target, caption_bank)
        delta = torch.zeros_like(source)
        global_momentum = torch.zeros_like(source)
        crop_momentum = torch.zeros_like(source)
        history = []
        xai_history = []

        # redundant - debuging
        write_json(sample_dir / 'attack_config.json', {
            'version':'6.4.1','xai_method':XAI_METHOD,'attention_layer':ATTENTION_LAYER,'steps':STEPS,'crop_area':CROP_AREA,'mode':'xai_all_steps',
            'alpha_global':ALPHA_GLOBAL,'alpha_crop':ALPHA_CROP,'epsilon':EPSILON,
            'momentum':MOMENTUM,'tau':TEMPERATURE,'omega':OMEGA,
            'merge':'crop completely replaces global inside box; next state is merged',
            'separate_momentum':True})
        write_json(sample_dir / "bank.json", caption_bank)
        write_json(sample_dir / "anchors.json", {
            "selected_target": selected,
            "target_caption": caption_bank["target_captions"][selected],
        })

        with torch.no_grad():
            caption_groups = ["source_captions", "target_captions", "negative_captions"]
            caption_vectors = {
                name: self.encoder.encode_text(caption_bank[name]).detach().cpu().numpy()
                for name in caption_groups
            }
            np.savez_compressed(sample_dir / "caption_embeddings.npz", **caption_vectors)
        step_dir = sample_dir / "steps"
        step_dir.mkdir(exist_ok=True)
        
        tensor_to_image(source).save(sample_dir / "source.png")
        tensor_to_image(target).save(sample_dir / "target.png")
        tensor_to_image(source).save(step_dir / "step_000.png")

        source_concepts = caption_bank["source_concepts"]
        target_concepts = caption_bank["target_concepts"]
        _, _ = self.save_xai(target, target_concepts, sample_dir, "target", 0)
        heat, initial_xai = self.save_xai(source, source_concepts, sample_dir, "source", 0)
        xai_history.append(initial_xai)
        if heat is None:
            raise RuntimeError("No valid source XAI map; cannot start guided cropping")

        # At step 0 the adversarial image equals the source image.
        np.savez_compressed(sample_dir / "xai" / "adversarial_000.npz", aggregate=heat.numpy())
        write_json(sample_dir / "xai" / "adversarial_000.json", {
            "step": 0,
            "role": "adversarial",
            "valid": True,
            "initially_equals_source": True,
            "dispersion": XAIMap.dispersion(heat),
        })

        attention_role = 'source'
        attention_step = 0
        latest_refresh_step = 0

        for step in tqdm(range(1, STEPS + 1), desc=f"CARA v6.4.1 sample {sample_id}"):
            if hasattr(self, "check_limits"):
                self.check_limits()
            # Always read the latest merged perturbation, for both views.
            top, left, height, width, mode = self.selector.select(heat, step)
            box = (top,left,height,width)
            if step==1 or step%SAVE_CROP_EVERY==0:
                self.snapshot_crop((source+delta).clamp(0,255),heat,box,sample_dir,step,
                                   attention_role,attention_step,latest_refresh_step)
            global_gradient, crop_gradient, metrics=self.loss_function.gradients(self.encoder, source, delta, box, positive, negatives)
            delta, global_momentum, crop_momentum = self.overwrite_updates(source, delta, global_momentum, crop_momentum, box, global_gradient,crop_gradient)
            
            history.append({'step':step, **metrics, 'mode':mode,
                            'crop':list(box), 'crop_area':CROP_AREA,
                            'update_rule':'global_then_crop_replace',
                            'attention_role':attention_role,'attention_step':attention_step,
                            'attention_reused':latest_refresh_step!=attention_step,
                            'global_updated_pixels':source.shape[-2]*source.shape[-1],
                            'crop_replaced_pixels':height*width})
            if step == 1 and hasattr(self, "first_step_gate"):
                write_json(sample_dir / "first_step.json", {
                    "history": history, "query_shape": [1, int(positive.shape[1])],
                    "positive_shape": list(positive.shape), "negative_shape": list(negatives.shape),
                    "global_gradient_shape": list(global_gradient.shape),
                    "crop_gradient_shape": list(crop_gradient.shape), "seed": SEED + sample_id,
                    "loss_formula": "score=-mean cross_entropy([omega*cos(q,p),q@N.T]/tau,0); ascent",
                    "direction": "independent branch ascent; crop overwrite; sum CE diagnostic only"})
                self.first_step_gate(sample_id)
            del global_gradient , crop_gradient

            if step % SAVE_EVERY == 0:
                current = (source + delta).clamp(0, 255)
                tensor_to_image(current).save(step_dir / f"step_{step:03d}.png")
                if SAVE_BRANCH_IMAGES:
                    global_dir=sample_dir/'global_steps';crop_dir=sample_dir/'crop_steps'
                    global_dir.mkdir(exist_ok=True);crop_dir.mkdir(exist_ok=True)
                    tensor_to_image(source+self.last_proposals['global']).save(global_dir/f'step_{step:03d}.png')
                    tensor_to_image(source[:,:,top:top+height,left:left+width]+self.last_proposals['crop']).save(
                        crop_dir/f'step_{step:03d}.png')
                heat_new, info = self.save_xai(current, source_concepts,
                                               sample_dir, "adversarial", step, previous=heat)
                latest_refresh_step=step
                if info.get('valid'):
                    attention_role='adversarial';attention_step=step
                xai_history.append(info)
                if heat_new is not None:
                    heat = heat_new
                write_json(sample_dir / "progress.json", {"history": history, "xai": xai_history})
                print(f"Step {step:03d} | global CE: {metrics['global_loss']:.4f} | "
                      f"crop CE: {metrics['crop_loss']:.4f} | overwrite: {mode}")

        result = (source + delta).clamp(0, 255)
        tensor_to_image(result).save(sample_dir / "adversarial.png")
        difference = (result.detach() - source).abs().max().item()
        if difference > EPSILON + 1e-5:
            raise AssertionError(f"L-infinity bound violated: {difference}")
        
        # write json
        write_json(sample_dir / "summary.json", {
            "sample_id": sample_id,
            "updates": STEPS,
            "method":"v6.4.1_qwen_global_then_crop_replace",
            "crop_area":CROP_AREA,
            "alpha_global":ALPHA_GLOBAL, "alpha_crop":ALPHA_CROP,
            "final_global_loss_before_update":history[-1]['global_loss'],
            "final_crop_loss_before_update":history[-1]['crop_loss'],
            "linf_pixels": difference,
            "target_caption": caption_bank["target_captions"][selected],
        })
        return result.detach().cpu()
