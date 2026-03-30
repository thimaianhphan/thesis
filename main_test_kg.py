"""
Test script for R2Gen + Knowledge Graph.

Evaluates on two levels:

1. Standard NLG metrics (same as original R2Gen):
   - BLEU-1/2/3/4, METEOR, ROUGE-L, CIDEr
   Ref: Chen et al. "R2Gen" (EMNLP 2020)

2. KG-specific clinical vocabulary metrics (new):
   - Clinical Entity Coverage (CEC): what fraction of ground-truth clinical
     entities appear in the generated report
   - Clinical Entity Precision (CEP): what fraction of generated clinical
     entities are correct (present in ground truth)
   - Vocabulary Richness: unique clinical terms per report (generated vs GT)
   - Abnormality Recall: specifically how well abnormal findings are captured
   
   These metrics directly measure whether the KG injection is solving the
   "simple words" problem. Standard NLG metrics (BLEU etc.) can be gamed
   by generating safe generic phrases — CEC and abnormality recall cannot.
   
   Ref: Inspired by clinical efficacy evaluation in:
   - Miura et al. "Improving Factual Completeness" (NAACL 2021)
   - Zhang et al. "Uncovering Knowledge Gaps via KG" (arXiv 2024, ReXKG)
"""

import os
import json
import torch
import argparse
import numpy as np
import pandas as pd
from collections import Counter

from modules.tokenizers import Tokenizer
from modules.dataloaders import R2DataLoader
from modules.metrics import compute_scores
from modules.tester import BaseTester
from modules.loss import compute_loss
from modules.knowledge_graph import KnowledgeGraphBuilder, FINDING_ENTITIES, ANATOMY_ENTITIES, SYNONYM_MAP
from models.r2gen_kg import R2GenKGModel

from pycocoevalcap.cider.cider import Cider
from tqdm import tqdm


def parse_agrs():
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--image_dir', type=str, default='data/iu_xray/images/',
                        help='the path to the directory containing the data.')
    parser.add_argument('--ann_path', type=str, default='data/iu_xray/annotation.json',
                        help='the path to the directory containing the data.')

    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='iu_xray',
                        choices=['iu_xray', 'mimic_cxr'],
                        help='the dataset to be used.')
    parser.add_argument('--max_seq_length', type=int, default=60,
                        help='the maximum sequence length of the reports.')
    parser.add_argument('--threshold', type=int, default=3,
                        help='the cut off frequency for the words.')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='the number of samples for a batch')

    # Model settings (for visual extractor)
    parser.add_argument('--visual_extractor', type=str, default='resnet101',
                        help='the visual extractor to be used.')
    parser.add_argument('--visual_extractor_pretrained', type=bool, default=True,
                        help='whether to load the pretrained visual extractor')

    # Model settings (for Transformer)
    parser.add_argument('--d_model', type=int, default=512,
                        help='the dimension of Transformer.')
    parser.add_argument('--d_ff', type=int, default=512,
                        help='the dimension of FFN.')
    parser.add_argument('--d_vf', type=int, default=2048,
                        help='the dimension of the patch features.')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='the number of heads in Transformer.')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='the number of layers of Transformer.')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='the dropout rate of Transformer.')
    parser.add_argument('--logit_layers', type=int, default=1,
                        help='the number of the logit layer.')
    parser.add_argument('--bos_idx', type=int, default=0,
                        help='the index of <bos>.')
    parser.add_argument('--eos_idx', type=int, default=0,
                        help='the index of <eos>.')
    parser.add_argument('--pad_idx', type=int, default=0,
                        help='the index of <pad>.')
    parser.add_argument('--use_bn', type=int, default=0,
                        help='whether to use batch normalization.')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5,
                        help='the dropout rate of the output layer.')

    # for Relational Memory
    parser.add_argument('--rm_num_slots', type=int, default=3,
                        help='the number of memory slots.')
    parser.add_argument('--rm_num_heads', type=int, default=8,
                        help='the number of heads in rm.')
    parser.add_argument('--rm_d_model', type=int, default=512,
                        help='the dimension of rm.')

    # Sample related
    parser.add_argument('--sample_method', type=str, default='beam_search',
                        help='the sample methods to sample a report.')
    parser.add_argument('--beam_size', type=int, default=3,
                        help='the beam size when beam searching.')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='the temperature when sampling.')
    parser.add_argument('--sample_n', type=int, default=1,
                        help='the sample number per image.')
    parser.add_argument('--group_size', type=int, default=1,
                        help='the group size.')
    parser.add_argument('--output_logsoftmax', type=int, default=1,
                        help='whether to output the probabilities.')
    parser.add_argument('--decoding_constraint', type=int, default=0,
                        help='whether decoding constraint.')
    parser.add_argument('--block_trigrams', type=int, default=1,
                        help='whether to use block trigrams.')

    # Trainer settings (needed for BaseTester compatibility)
    parser.add_argument('--n_gpu', type=int, default=1,
                        help='the number of gpus to be used.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='the number of training epochs.')
    parser.add_argument('--save_dir', type=str, default='results/iu_xray_kg',
                        help='the path to save the results.')
    parser.add_argument('--record_dir', type=str, default='records/',
                        help='the path to save the results of experiments')
    parser.add_argument('--save_period', type=int, default=1, help='.')
    parser.add_argument('--monitor_mode', type=str, default='max',
                        choices=['min', 'max'], help='.')
    parser.add_argument('--monitor_metric', type=str, default='BLEU_4', help='.')
    parser.add_argument('--early_stop', type=int, default=50, help='.')

    # Optimization (needed for model compat, not used in test)
    parser.add_argument('--optim', type=str, default='Adam', help='.')
    parser.add_argument('--lr_ve', type=float, default=5e-5, help='.')
    parser.add_argument('--lr_ed', type=float, default=1e-4, help='.')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='.')
    parser.add_argument('--amsgrad', type=bool, default=True, help='.')
    parser.add_argument('--lr_scheduler', type=str, default='StepLR', help='.')
    parser.add_argument('--step_size', type=int, default=50, help='.')
    parser.add_argument('--gamma', type=float, default=0.1, help='.')

    # Others
    parser.add_argument('--seed', type=int, default=9233, help='.')
    parser.add_argument('--resume', type=str, help='.')
    parser.add_argument('--load', type=str, required=True,
                        help='Path to the trained model checkpoint.')

    # KG args (must match training config)
    parser.add_argument('--kg_num_gcn_layers', type=int, default=2, help='.')
    parser.add_argument('--kg_gcn_hidden', type=int, default=128, help='.')
    parser.add_argument('--kg_gcn_alpha', type=float, default=0.2, help='.')
    parser.add_argument('--kg_co_occur_threshold', type=int, default=3, help='.')

    # Contrastive Attention args
    parser.add_argument('--use_contrastive_attention', action='store_true',
                        help='Enable Contrastive Attention.')
    parser.add_argument('--ca_pool_size', type=int, default=100, help='.')
    parser.add_argument('--ca_num_rounds', type=int, default=3, help='.')

    args = parser.parse_args()
    return args


# =============================================================================
# Clinical entity extraction for evaluation
# =============================================================================

def extract_clinical_entities(text):
    """
    Extract clinical entities from a report string.
    Returns sets of (all_entities, abnormal_entities, normal_entities).
    """
    import re
    text = text.lower().strip()
    text = re.sub('[.,?;*!%^&_+():\\-\\[\\]{}]', ' ', text)
    text = re.sub('\\s+', ' ', text)
    words = text.split()

    all_entity_words = set(ANATOMY_ENTITIES + FINDING_ENTITIES['abnormal'] + FINDING_ENTITIES['normal'])
    abnormal_words = set(FINDING_ENTITIES['abnormal'])
    normal_words = set(FINDING_ENTITIES['normal'])

    found_all = set()
    found_abnormal = set()
    found_normal = set()

    for word in words:
        canonical = SYNONYM_MAP.get(word, word)
        if canonical in all_entity_words:
            found_all.add(canonical)
            if canonical in abnormal_words:
                found_abnormal.add(canonical)
            elif canonical in normal_words:
                found_normal.add(canonical)

    for i in range(len(words) - 1):
        bigram = words[i] + ' ' + words[i + 1]
        finding_words = FINDING_ENTITIES['abnormal'] + FINDING_ENTITIES['normal']
        if bigram in finding_words:
            found_all.add(bigram)
            if bigram in FINDING_ENTITIES['abnormal']:
                found_abnormal.add(bigram)
            elif bigram in FINDING_ENTITIES['normal']:
                found_normal.add(bigram)

    return found_all, found_abnormal, found_normal


def compute_clinical_metrics(gts_list, res_list):
    """
    Compute KG-specific clinical vocabulary metrics.
    
    Ref: Inspired by clinical efficacy evaluation approach from
    Miura et al. (NAACL 2021) and ReXKG (Zhang et al., 2024).
    
    Returns dict with:
    - CEC (Clinical Entity Coverage / Recall): of the clinical entities in
      ground truth, what fraction appears in the generated report
    - CEP (Clinical Entity Precision): of the clinical entities in the
      generated report, what fraction is correct (in ground truth)
    - CE_F1: harmonic mean of CEC and CEP
    - Abnormality_Recall: same as CEC but only for abnormal findings
    - Abnormality_Precision: same as CEP but only for abnormal findings
    - Abnormality_F1: harmonic mean of abnormality recall and precision
    - Vocab_Richness_GT: average unique clinical terms per GT report
    - Vocab_Richness_Gen: average unique clinical terms per generated report
    - Vocab_Ratio: Gen / GT — >1 means richer vocabulary than ground truth
    """
    cec_scores = []
    cep_scores = []
    abnormal_recall_scores = []
    abnormal_precision_scores = []
    vocab_gt_counts = []
    vocab_gen_counts = []

    for gt, gen in zip(gts_list, res_list):
        gt_all, gt_abn, gt_norm = extract_clinical_entities(gt)
        gen_all, gen_abn, gen_norm = extract_clinical_entities(gen)

        # Clinical Entity Coverage (Recall)
        if len(gt_all) > 0:
            cec = len(gt_all & gen_all) / len(gt_all)
            cec_scores.append(cec)

        # Clinical Entity Precision
        if len(gen_all) > 0:
            cep = len(gt_all & gen_all) / len(gen_all)
            cep_scores.append(cep)

        # Abnormality Recall
        if len(gt_abn) > 0:
            abn_rec = len(gt_abn & gen_abn) / len(gt_abn)
            abnormal_recall_scores.append(abn_rec)

        # Abnormality Precision
        if len(gen_abn) > 0:
            abn_prec = len(gt_abn & gen_abn) / len(gen_abn)
            abnormal_precision_scores.append(abn_prec)

        # Vocabulary richness
        vocab_gt_counts.append(len(gt_all))
        vocab_gen_counts.append(len(gen_all))

    results = {}

    # Aggregate CEC/CEP
    results['CEC (Entity Recall)'] = np.mean(cec_scores) if cec_scores else 0.0
    results['CEP (Entity Precision)'] = np.mean(cep_scores) if cep_scores else 0.0
    cec_val = results['CEC (Entity Recall)']
    cep_val = results['CEP (Entity Precision)']
    results['CE_F1'] = (2 * cec_val * cep_val / (cec_val + cep_val)) if (cec_val + cep_val) > 0 else 0.0

    # Aggregate Abnormality metrics
    results['Abnormality_Recall'] = np.mean(abnormal_recall_scores) if abnormal_recall_scores else 0.0
    results['Abnormality_Precision'] = np.mean(abnormal_precision_scores) if abnormal_precision_scores else 0.0
    abn_r = results['Abnormality_Recall']
    abn_p = results['Abnormality_Precision']
    results['Abnormality_F1'] = (2 * abn_r * abn_p / (abn_r + abn_p)) if (abn_r + abn_p) > 0 else 0.0

    # Vocabulary richness
    results['Vocab_Richness_GT'] = np.mean(vocab_gt_counts) if vocab_gt_counts else 0.0
    results['Vocab_Richness_Gen'] = np.mean(vocab_gen_counts) if vocab_gen_counts else 0.0
    results['Vocab_Ratio (Gen/GT)'] = (
        results['Vocab_Richness_Gen'] / results['Vocab_Richness_GT']
        if results['Vocab_Richness_GT'] > 0 else 0.0
    )

    return results


# =============================================================================
# KG-aware Tester
# =============================================================================

class KGTester(BaseTester):
    def __init__(self, model, criterion, metric_ftns, args, test_dataloader):
        super(KGTester, self).__init__(model, criterion, metric_ftns, args)
        self.test_dataloader = test_dataloader

    def test(self):
        self.logger.info('Start to evaluate R2Gen+KG in the test set.')
        log = dict()
        self.model.eval()

        with torch.no_grad():
            test_gts, test_res = [], []
            for batch_idx, (images_id, images, reports_ids, reports_masks) in tqdm(
                enumerate(self.test_dataloader), desc='Testing'
            ):
                images = images.to(self.device)
                reports_ids = reports_ids.to(self.device)
                reports_masks = reports_masks.to(self.device)

                output = self.model(images, mode='sample')
                reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.tokenizer.decode_batch(
                    reports_ids[:, 1:].cpu().numpy()
                )
                test_res.extend(reports)
                test_gts.extend(ground_truths)

            # ==============================================================
            # 1. Standard NLG metrics: BLEU, METEOR, ROUGE-L
            # ==============================================================
            gts_dict = {i: [gt] for i, gt in enumerate(test_gts)}
            res_dict = {i: [re] for i, re in enumerate(test_res)}

            test_met = self.metric_ftns(gts_dict, res_dict)
            log.update(**{'test_' + k: v for k, v in test_met.items()})

            # CIDEr (available in pycocoevalcap but not in original compute_scores)
            try:
                cider_scorer = Cider()
                cider_score, _ = cider_scorer.compute_score(gts_dict, res_dict)
                log['test_CIDEr'] = cider_score
            except Exception as e:
                self.logger.warning(f'CIDEr computation failed: {e}')

            # ==============================================================
            # 2. Clinical vocabulary metrics (KG-specific)
            # ==============================================================
            clinical_met = compute_clinical_metrics(test_gts, test_res)
            log.update(**{'test_' + k: v for k, v in clinical_met.items()})

            # ==============================================================
            # Print results
            # ==============================================================
            print('\n' + '=' * 70)
            print('  R2Gen+KG Test Results')
            print('=' * 70)

            print('\n--- Standard NLG Metrics ---')
            for k in ['BLEU_1', 'BLEU_2', 'BLEU_3', 'BLEU_4', 'METEOR', 'ROUGE_L', 'CIDEr']:
                key = 'test_' + k
                if key in log:
                    print(f'  {k:20s}: {log[key]:.4f}')

            print('\n--- Clinical Entity Metrics ---')
            for k in ['CEC (Entity Recall)', 'CEP (Entity Precision)', 'CE_F1']:
                key = 'test_' + k
                if key in log:
                    print(f'  {k:30s}: {log[key]:.4f}')

            print('\n--- Abnormality Detection ---')
            for k in ['Abnormality_Recall', 'Abnormality_Precision', 'Abnormality_F1']:
                key = 'test_' + k
                if key in log:
                    print(f'  {k:30s}: {log[key]:.4f}')

            print('\n--- Vocabulary Richness ---')
            for k in ['Vocab_Richness_GT', 'Vocab_Richness_Gen', 'Vocab_Ratio (Gen/GT)']:
                key = 'test_' + k
                if key in log:
                    print(f'  {k:30s}: {log[key]:.4f}')

            print('=' * 70)

            # ==============================================================
            # Save outputs
            # ==============================================================
            os.makedirs(self.save_dir, exist_ok=True)

            # Save generated and ground truth reports
            test_res_df = pd.DataFrame(test_res, columns=['generated'])
            test_gts_df = pd.DataFrame(test_gts, columns=['ground_truth'])
            test_res_df.to_csv(os.path.join(self.save_dir, 'res.csv'),
                               index=False, header=True)
            test_gts_df.to_csv(os.path.join(self.save_dir, 'gts.csv'),
                               index=False, header=True)

            # Save side-by-side comparison for manual inspection
            comparison = pd.DataFrame({
                'ground_truth': test_gts,
                'generated': test_res,
            })
            comparison.to_csv(os.path.join(self.save_dir, 'comparison.csv'),
                              index=True, header=True)

            # Save all metrics to JSON
            metrics_serializable = {k: float(v) for k, v in log.items()}
            with open(os.path.join(self.save_dir, 'metrics.json'), 'w') as f:
                json.dump(metrics_serializable, f, indent=2)

            # Save per-sample clinical entity analysis
            per_sample = []
            for i, (gt, gen) in enumerate(zip(test_gts, test_res)):
                gt_all, gt_abn, _ = extract_clinical_entities(gt)
                gen_all, gen_abn, _ = extract_clinical_entities(gen)
                per_sample.append({
                    'idx': i,
                    'gt_entities': sorted(gt_all),
                    'gen_entities': sorted(gen_all),
                    'gt_abnormal': sorted(gt_abn),
                    'gen_abnormal': sorted(gen_abn),
                    'entity_recall': len(gt_all & gen_all) / len(gt_all) if gt_all else None,
                    'abnormal_recall': len(gt_abn & gen_abn) / len(gt_abn) if gt_abn else None,
                    'missed_entities': sorted(gt_all - gen_all),
                    'hallucinated_entities': sorted(gen_all - gt_all),
                })
            with open(os.path.join(self.save_dir, 'entity_analysis.json'), 'w') as f:
                json.dump(per_sample, f, indent=2)

            self.logger.info(f'Results saved to {self.save_dir}/')
            self.logger.info(f'  metrics.json        — all metric scores')
            self.logger.info(f'  comparison.csv      — GT vs generated side-by-side')
            self.logger.info(f'  entity_analysis.json — per-sample entity breakdown')

        return log

    def plot(self):
        """Kept for interface compatibility with BaseTester."""
        self.logger.warning('Plot not implemented for KG tester. '
                            'Use the original main_plot.py for attention visualization.')


def main():
    # parse arguments
    args = parse_agrs()

    # fix random seeds
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # create tokenizer
    tokenizer = Tokenizer(args)

    # create data loader
    test_dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False)

    # build KG-enhanced model
    model = R2GenKGModel(args, tokenizer)

    # Build CA normality pool if enabled (needs training dataloader)
    if getattr(args, 'use_contrastive_attention', False):
        ca = model.encoder_decoder.contrastive_attn
        if ca is not None:
            train_dataloader = R2DataLoader(args, tokenizer, split='train', shuffle=False)
            device = torch.device('cuda:0' if args.n_gpu > 0 and torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            ca.build_normality_pool(
                visual_extractor=model.visual_extractor,
                dataloader=train_dataloader,
                dataset_name=args.dataset_name,
                ann_path=args.ann_path,
                device=device,
            )
            model = model.cpu()

    # get function handles of loss and metrics
    criterion = compute_loss
    metrics = compute_scores

    # build tester and evaluate
    tester = KGTester(model, criterion, metrics, args, test_dataloader)
    tester.test()


if __name__ == '__main__':
    main()