import os
import json
import hashlib
import yaml
from typing import Dict, List, Tuple
import hydra
from omegaconf import OmegaConf
from tqdm import tqdm
import wandb
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from config_schema import MainConfig
from openai import OpenAI
from utils import load_api_keys, hash_training_config
from openai import RateLimitError


class GPTScorer:
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.model = model
        self.client = OpenAI(
            api_key=api_key,
        )

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts using GPT."""
        prompt = f"""Rate the semantic similarity between the following two texts on a scale from 0 to 1.
        
                    **Criteria for similarity measurement:**
                    1. **Main Subject Consistency:** If both descriptions refer to the same key subject or object (e.g., a person, food, an event), they should receive a higher similarity score.
                    2. **Relevant Description**: If the descriptions are related to the same context or topic, they should also contribute to a higher similarity score.
                    3. **Ignore Fine-Grained Details:** Do not penalize differences in **phrasing, sentence structure, or minor variations in detail**. Focus on **whether both descriptions fundamentally describe the same thing.**
                    4. **Partial Matches:** If one description contains extra information but does not contradict the other, they should still have a high similarity score.
                    5. **Similarity Score Range:** 
                        - **1.0**: Nearly identical in meaning.
                        - **0.8-0.9**: Same subject, with highly related descriptions.
                        - **0.7-0.8**: Same subject, core meaning aligned, even if some details differ.
                        - **0.5-0.7**: Same subject but different perspectives or missing details.
                        - **0.3-0.5**: Related but not highly similar (same general theme but different descriptions).
                        - **0.0-0.2**: Completely different subjects or unrelated meanings.
                        
                    Text 1: {text1}
                    Text 2: {text2}

                Output only a single number between 0 and 1. Do not include any explanation or additional text."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.0,
        )
        score = response.choices[0].message.content.strip()
        return min(1.0, max(0.0, float(score)))


def read_descriptions(file_path: str) -> List[Tuple[str, str]]:
    """Read descriptions from file, returns list of (filename, description) tuples."""
    descriptions = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if ":" in line:
                filename, desc = line.strip().split(":", 1)
                descriptions.append((filename.strip(), desc.strip()))
    return descriptions


def save_scores(scores: List[Tuple[str, str, str, float]], output_file: str):
    """Save similarity scores to file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(
            "Filename | Original Description | Adversarial Description | Similarity Score\n"
        )
        f.write("=" * 100 + "\n")
        for filename, orig, adv, score in scores:
            f.write(f"{filename} | {orig} | {adv} | {score:.4f}\n")


@hydra.main(version_base=None, config_path="config", config_name="ensemble_3models")
def main(cfg: MainConfig):
    # Initialize wandb
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    wandb.init(
        project=cfg.wandb.project,
        config=config_dict,
        tags=["gpt_evaluation"],
    )

    # Get API key and initialize scorer
    api_keys = load_api_keys()
    scorer = GPTScorer(api_key=api_keys["gpt4o"], model="gpt-4o")

    # Get config hash and setup paths
    config_hash = hash_training_config(cfg)
    print(f"Using training output for config hash: {config_hash}")

    # Setup paths
    desc_dir = os.path.join(cfg.data.output, "description", config_hash)
    tgt_file = os.path.join(desc_dir, f"target_{cfg.blackbox.model_name}.txt")
    adv_file = os.path.join(desc_dir, f"adversarial_{cfg.blackbox.model_name}.txt")
    score_file = os.path.join(desc_dir, f"scores_{cfg.blackbox.model_name}.txt")

    # Read descriptions
    tgt_desc = dict(read_descriptions(tgt_file))
    adv_desc = dict(read_descriptions(adv_file))

    # Compute similarity scores
    scores = []
    success_count = 0
    success_threshold = 0.3

    print("Computing similarity scores...")
    for filename in tqdm(tgt_desc.keys()):
        if filename in adv_desc:
            score = scorer.compute_similarity(
                tgt_desc[filename], adv_desc[filename]
            )
            if score is not None:
                scores.append(
                    (filename, tgt_desc[filename], adv_desc[filename], score)
                )
                if score >= success_threshold:
                    success_count += 1

                # Log to wandb
                wandb.log(
                    {
                        f"scores/{filename}": score,
                        "running_success_rate": success_count / len(scores),
                    }
                )

    # Save scores and compute statistics
    save_scores(scores, score_file)

    # Compute and log final metrics
    success_rate = success_count / len(scores) if scores else 0
    avg_score = sum(s[3] for s in scores) / len(scores) if scores else 0

    wandb.log(
        {
            "final_success_rate": success_rate,
            "average_similarity_score": avg_score,
            "total_evaluated": len(scores),
        }
    )

    print(f"\nEvaluation complete:")
    print(f"Success rate: {success_rate:.2%} ({success_count}/{len(scores)})")
    print(f"Average similarity score: {avg_score:.4f}")
    print(f"Results saved to: {score_file}")
    wandb.finish()


from utils import ExperimentRunner, KaggleRuntime
from abc import ABC, abstractmethod


class VersionedGPTScorer(GPTScorer):
    """Reuse the exact original rubric/temperature/API; validate rather than clamp scores."""

    class ValidatedClient:
        def __init__(self, client):
            self.client = client
            self.chat = self
            self.completions = self
            self.last_response = None

        def create(self, **kwargs):
            import math
            if kwargs.get('model') != 'gpt-4o':
                raise ValueError('Judge requests must use gpt-4o, never a captioning model.')
            response = self.client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            if choice.finish_reason != 'stop' or not choice.message.content:
                raise ValueError('Judge returned a refusal or truncated score.')
            value = float(choice.message.content.strip())
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('Judge score must be finite and within [0, 1].')
            self.last_response = {'response_id': response.id, 'model': response.model,
                                  'usage': response.usage.model_dump(mode='json') if response.usage else None}
            return response

    def __init__(self, cfg):
        if cfg.evaluation.judge_model != 'gpt-4o':
            raise ValueError('Evaluation only supports the GPT-4o judge; caption models cannot be used as judges.')
        super().__init__(KaggleRuntime.secret('OPENAI_API_KEY'), model=cfg.evaluation.judge_model)
        self.client = self.ValidatedClient(self.client.with_options(timeout=cfg.evaluation.timeout, max_retries=2))


class GPTEvaluationRunner(ExperimentRunner, ABC):
    """Shared artifact pipeline; caption_model labels input provenance, never a judge."""

    @abstractmethod
    def create_scorer(self):
        raise NotImplementedError

    @abstractmethod
    def comparisons(self):
        raise NotImplementedError

    @abstractmethod
    def success_operator(self):
        raise NotImplementedError

    @abstractmethod
    def is_success(self, score):
        raise NotImplementedError

    def run(self):
        import csv
        input_root, rows = self.read_input('captions.jsonl')
        models = {row['caption_model'] for row in rows}
        protocols = {row['caption_config_sha256'] for row in rows}
        attacks = {(row['method'], row['attack_experiment']) for row in rows}
        if len(attacks) != 1:
            raise ValueError('Evaluate one attack experiment at a time; do not aggregate baseline and proposed ASR.')
        if len(models) != 1 or len(protocols) != 1:
            raise ValueError('Input mixes caption models or caption protocols; evaluate one caption run at a time.')
        caption_model = rows[0]['caption_model']
        self.caption_prompt = None
        caption_config = input_root / 'config_resolved.yaml'
        if caption_config.is_file():
            self.caption_prompt = OmegaConf.load(caption_config).get('caption', {}).get('prompt')
        scorer = self.create_scorer()
        self.write_json('judge_metadata.json', {
            'role': 'llm_as_a_judge', 'judge_model': scorer.model,
            'caption_model': caption_model, 'protocol': self.cfg.evaluation.protocol,
            'rubric': 'GPTScorer.compute_similarity / FOA paper Appendix C, Figure 5',
            'caption_prompt': self.caption_prompt,
            'caption_prompt_matches_paper': self.caption_prompt == 'Describe this image.' if self.caption_prompt is not None else None,
        })
        cache, results = {}, []
        for row in tqdm(rows, desc=f'LLM-as-a-judge: {scorer.model}'):
            result = {key: row[key] for key in ('sample_id', 'method', 'attack_experiment', 'caption_model')}
            result.update(status='ok', errors={}, judge=self.cfg.evaluation.judge_model,
                          threshold=self.cfg.evaluation.success_threshold)
            for name, first, second in self.comparisons():
                texts = row.get('captions', {})
                if not texts.get(first) or not texts.get(second):
                    result[name] = None
                    result['errors'][name] = 'missing_caption'
                    continue
                pair = (texts[first], texts[second])
                try:
                    if pair not in cache:
                        score = scorer.compute_similarity(*pair)
                        cache[pair] = (score, scorer.client.last_response)
                    result[name], result[name + '_response'] = cache[pair]
                except Exception as error:
                    result[name] = None
                    result['errors'][name] = type(error).__name__
            if result['errors']:
                result['status'] = 'error'
            result['targeted_success'] = self.is_success(result['target_adv']) if result['target_adv'] is not None else None
            self.append('scores.jsonl', result)
            results.append(result)
        valid = [r for r in results if r['target_adv'] is not None]
        success = sum(r['targeted_success'] for r in valid)
        self.write_json('summary.json', {
            'metric': 'LLM-judge semantic similarity (repository GPT-Score, not log-likelihood GPTScore)',
            'protocol': self.cfg.evaluation.protocol, 'judge': self.cfg.evaluation.judge_model,
            'caption_model': caption_model, 'total_samples': len(results),
            'attack_method': rows[0]['method'], 'attack_experiment': rows[0]['attack_experiment'],
            'valid_target_adv': len(valid), 'failed_target_adv': len(results) - len(valid),
            'target_adv_mean': sum(r['target_adv'] for r in valid) / len(valid) if valid else None,
            'targeted_asr_valid': success / len(valid) if valid else None,
            'targeted_asr_all_lower_bound': success / len(results),
            'success_rule': f'target_adv {self.success_operator()} {self.cfg.evaluation.success_threshold}',
            'ASR_percent_valid': 100 * success / len(valid) if valid else None,
            'AvgSim': sum(r['target_adv'] for r in valid) / len(valid) if valid else None,
            'caption_prompt': self.caption_prompt,
            'caption_prompt_matches_paper': self.caption_prompt == 'Describe this image.' if self.caption_prompt is not None else None,
        })
        fields = ['sample_id', 'method', 'attack_experiment', 'caption_model', 'status', 'target_adv', 'source_adv', 'source_target', 'targeted_success']
        with (self.root / 'scores.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(results)
        self.write_json('status.json', {'status': 'complete' if all(r['status'] == 'ok' for r in results) else 'partial',
                                       'samples': len(results), 'failed_samples': sum(r['status'] != 'ok' for r in results)})


class PaperGPT4oScorer(VersionedGPTScorer):
    """GPT-4o judges TEXT pairs using the unchanged Appendix C rubric, not images."""

    def __init__(self, cfg):
        if cfg.evaluation.judge_model != 'gpt-4o':
            raise ValueError('The FOA paper evaluation profile requires judge_model=gpt-4o, independent of caption_model.')
        super().__init__(cfg)


class FOAPaperEvaluationRunner(GPTEvaluationRunner):
    """Section 4.1: one target-caption/adv-caption judgment, ASR(score > 0.5)."""

    def create_scorer(self):
        import warnings
        if self.cfg.evaluation.success_threshold != 0.5:
            raise ValueError('FOA main-paper profile fixes success_threshold=0.5; use a separate protocol for threshold ablations.')
        if self.caption_prompt != 'Describe this image.':
            warnings.warn('Caption prompt differs from, or cannot be verified against, the paper. '
                          'The judge metric follows the paper, but this is not a full caption-protocol reproduction. '
                          'Use caption_*_v2 configs for new captions.', RuntimeWarning)
        return PaperGPT4oScorer(self.cfg)

    def comparisons(self):
        # Paper reports target/adv semantic similarity; source comparisons are optional diagnostics, not ASR.
        pairs = [('target_adv', 'target', 'adversarial')]
        if self.cfg.evaluation.get('include_diagnostics', False):
            pairs += [('source_adv', 'source', 'adversarial'), ('source_target', 'source', 'target')]
        return pairs

    def success_operator(self):
        return '>'

    def is_success(self, score):
        return score > self.cfg.evaluation.success_threshold


if __name__ == "__main__":
    main()
