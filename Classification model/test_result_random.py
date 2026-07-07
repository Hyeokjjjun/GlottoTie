import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import os
from tqdm import tqdm
import json

from config import config
from dataloader import OurDeviceDataset
from model import Encoder, PrototypicalNetwork


class RandomTrialEvaluator:
    """Randomly sample the support set N times to compute the accuracy distribution (mean +/- SD).

    - No best-of-N selection is performed. All trial results are reported as-is.
    - The confusion matrix is aggregated by accumulating the query predictions over all trials.
    - Support is drawn with replace=False and query is the remainder excluding support,
      so there is no support<->query overlap (leakage) within a single trial.
    """

    def __init__(self, model, dataset, k_shot=5, n_trials=100):
        self.model = model
        self.dataset = dataset
        self.k_shot = k_shot
        self.n_trials = n_trials
        self.device = config.DEVICE

    def sample_support_query(self):
        """Sample support and query (all remaining items excluding support become query)"""
        support_data = []
        support_labels = []
        query_data = []
        query_labels = []
        support_indices = {}

        for class_idx in range(3):
            files = self.dataset.class_to_files[class_idx]
            n_samples = len(files)

            # Select k_shot items without replacement (replace=False) -> no support<->query overlap
            support_idx = np.random.choice(n_samples, self.k_shot, replace=False)
            support_indices[class_idx] = support_idx.tolist()

            for idx in support_idx:
                data = self.dataset.load_data(files[idx])
                support_data.append(data)
                support_labels.append(class_idx)

            for idx, file_path in enumerate(files):
                if idx not in support_idx:
                    data = self.dataset.load_data(file_path)
                    query_data.append(data)
                    query_labels.append(class_idx)

        support = torch.stack(support_data)
        support_lbl = torch.LongTensor(support_labels)
        query = torch.stack(query_data)
        query_lbl = torch.LongTensor(query_labels)

        return support, support_lbl, query, query_lbl, support_indices

    def evaluate_single_trial(self):
        """Evaluate a single trial"""
        support, support_labels, query, query_labels, support_indices = \
            self.sample_support_query()

        support = support.to(self.device)
        support_labels = support_labels.to(self.device)
        query = query.to(self.device)
        query_labels = query_labels.to(self.device)

        with torch.no_grad():
            logits = self.model(support, support_labels, query,
                                n_way=3, k_shot=self.k_shot)
            _, predicted = torch.max(logits, 1)
            accuracy = (predicted == query_labels).float().mean().item()

        return {
            'accuracy': accuracy,
            'support': support.cpu(),
            'support_labels': support_labels.cpu(),
            'query': query.cpu(),
            'query_labels': query_labels.cpu(),
            'predicted': predicted.cpu(),
            'logits': logits.cpu(),
            'support_indices': support_indices
        }

    def run_trials(self):
        """Run all N trials and return the result list + statistics (no selection)"""
        print(f"\nRunning {self.n_trials} random-support trials "
              f"(K-shot={self.k_shot}, no best-of-N selection)...")

        all_results = []
        for trial in tqdm(range(self.n_trials), desc="Sampling"):
            result = self.evaluate_single_trial()
            result['trial_id'] = trial
            all_results.append(result)

        accs = np.array([r['accuracy'] for r in all_results]) * 100.0
        n = len(accs)
        mean = float(accs.mean())
        std = float(accs.std(ddof=1)) if n > 1 else 0.0
        sem = std / np.sqrt(n) if n > 1 else 0.0
        ci95 = 1.96 * sem  # Half-width of the 95% confidence interval of the mean

        stats = {
            'n_trials': int(n),
            'k_shot': int(self.k_shot),
            'mean_accuracy': mean,
            'std_accuracy': std,
            'sem_accuracy': float(sem),
            'ci95_halfwidth': float(ci95),
            'ci95_lower': float(mean - ci95),
            'ci95_upper': float(mean + ci95),
            'min_accuracy': float(accs.min()),
            'max_accuracy': float(accs.max()),
            'median_accuracy': float(np.median(accs)),
            'all_accuracies': accs.tolist(),
        }

        print(f"\n=== Accuracy over {n} random-support trials (K-shot={self.k_shot}) ===")
        print(f"  Mean ± SD : {mean:.2f} ± {std:.2f} %")
        print(f"  95% CI    : [{mean - ci95:.2f}, {mean + ci95:.2f}] %")
        print(f"  Min / Max : {accs.min():.2f} / {accs.max():.2f} %")
        print(f"  Median    : {np.median(accs):.2f} %")

        return all_results, stats


class ResultVisualizer:
    """Result visualization (based on mean +/- SD)"""

    def __init__(self, save_dir='./results_3way_5shot'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(f'{save_dir}/confusion_matrices', exist_ok=True)
        os.makedirs(f'{save_dir}/tsne', exist_ok=True)

    def save_accuracy_stats(self, stats):
        """Save accuracy statistics (mean +/- SD, 95% CI, full distribution)"""
        with open(f'{self.save_dir}/accuracy_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Saved: {self.save_dir}/accuracy_stats.json")

    def plot_accuracy_distribution(self, stats):
        """Histogram of the accuracy distribution over all trials (evidence of honest reporting)"""
        accs = np.array(stats['all_accuracies'])
        mean, std = stats['mean_accuracy'], stats['std_accuracy']

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(accs, bins=20, color='#4ECDC4', edgecolor='black', alpha=0.8)
        ax.axvline(mean, color='red', linestyle='--', linewidth=2,
                   label=f'Mean = {mean:.2f}%')
        ax.axvspan(mean - std, mean + std, color='red', alpha=0.12,
                   label=f'±1 SD = {std:.2f}%')
        ax.set_xlabel('Accuracy (%)', fontsize=12)
        ax.set_ylabel('Number of trials', fontsize=12)
        ax.set_title(f"Accuracy Distribution over {stats['n_trials']} Random-Support Trials "
                     f"(k={stats['k_shot']})",
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/accuracy_distribution.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.save_dir}/accuracy_distribution.png")

    def plot_aggregated_confusion_matrix(self, all_results):
        """Confusion matrix accumulating query predictions over all N trials (no selection)"""
        class_names = ['Class 0', 'Class 1', 'Class 2']

        all_true, all_pred = [], []
        for result in all_results:
            all_true.extend(result['query_labels'].numpy())
            all_pred.extend(result['predicted'].numpy())

        cm = confusion_matrix(all_true, all_pred, labels=[0, 1, 2])
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        cm_data = {
            'aggregated_over_all_trials': True,
            'n_trials': len(all_results),
            'confusion_matrix_counts': cm.tolist(),
            'confusion_matrix_normalized': cm_norm.tolist(),
            'class_names': class_names
        }
        with open(f'{self.save_dir}/confusion_matrices/aggregated_cm.json', 'w') as f:
            json.dump(cm_data, f, indent=2)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap='Greens',
                    xticklabels=class_names, yticklabels=class_names,
                    ax=ax, cbar_kws={'label': 'Percentage (%)'},
                    vmin=0, vmax=100)
        ax.set_title(f"Aggregated Confusion Matrix ({len(all_results)} trials)",
                     fontsize=14, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/confusion_matrices/aggregated_cm.png',
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.save_dir}/confusion_matrices/aggregated_cm.(png/json)")

    def plot_tsne(self, model, representative_case, dataset):
        """t-SNE visualization -- uses the 'representative' trial closest to the mean (not the best case)"""
        support = representative_case['support'].to(config.DEVICE)
        query = representative_case['query'].to(config.DEVICE)
        support_labels = representative_case['support_labels'].numpy()
        query_labels = representative_case['query_labels'].numpy()

        with torch.no_grad():
            support_emb = model.encoder(support).cpu().numpy()
            query_emb = model.encoder(query).cpu().numpy()

        all_emb = np.vstack([support_emb, query_emb])
        all_labels = np.hstack([support_labels, query_labels])
        all_types = ['Support'] * len(support_labels) + ['Query'] * len(query_labels)

        # perplexity must be smaller than the number of samples -> auto-adjust (handles small-sample groups)
        n_points = all_emb.shape[0]
        perplexity = max(2, min(30, (n_points - 1) // 3))

        # Skip t-SNE if there are too few samples (e.g., a 2-shot group)
        if n_points < 4:
            print(f"  [skip] t-SNE: too few points (n={n_points})")
            return

        print(f"  Computing t-SNE (perplexity={perplexity}, n={n_points})...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        embeddings_2d = tsne.fit_transform(all_emb)

        tsne_data = {
            'representative_trial_id': int(representative_case['trial_id']),
            'representative_accuracy': float(representative_case['accuracy'] * 100),
            'perplexity': int(perplexity),
            'embeddings_2d': embeddings_2d.tolist(),
            'labels': all_labels.tolist(),
            'types': all_types,
            'support_indices': list(range(len(support_labels))),
            'query_indices': list(range(len(support_labels), len(all_labels)))
        }
        with open(f'{self.save_dir}/tsne/tsne_data.json', 'w') as f:
            json.dump(tsne_data, f, indent=2)

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        markers_dict = {'Support': 's', 'Query': 'o'}

        for class_idx in range(3):
            for data_type in ['Support', 'Query']:
                mask = (all_labels == class_idx) & (np.array(all_types) == data_type)
                if mask.sum() > 0:
                    ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                               c=[colors[class_idx]], marker=markers_dict[data_type],
                               s=150 if data_type == 'Support' else 100,
                               alpha=0.8 if data_type == 'Support' else 0.6,
                               edgecolors='black',
                               linewidth=1.5 if data_type == 'Support' else 1,
                               label=f'Class {class_idx} ({data_type})')

        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax.set_title(f"t-SNE Visualization (Representative Trial, "
                     f"Acc: {representative_case['accuracy']*100:.1f}%)",
                     fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/tsne/tsne_2d.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {self.save_dir}/tsne/tsne_2d.(png/json)")


def pick_representative_case(all_results, mean_acc_percent):
    """Pick the trial closest to the mean accuracy as the representative case (not the best)"""
    target = mean_acc_percent / 100.0
    return min(all_results, key=lambda r: abs(r['accuracy'] - target))


# ========================================================================== #
#  Run configuration  (bimodal / full 60-channel baseline)
# ========================================================================== #
DATA_ROOT = '/test_data'
CKPT_PATH = './checkpoints/best_model.pth'   # bimodal (full) model checkpoint


# (data folder name, result suffix, k_shot, n_trials)
DATASETS = [
    # folder,                         suffix,               k_shot, n_trials
    ('test_data_custom',           'custom',              2, 50),
    ('test_data_commercial',       'commercial',          2, 50),
]


def process_dataset(model, data_folder, result_suffix, k_shot, n_trials):
    """Mean +/- SD evaluation + visualization for a single dataset"""
    data_path = os.path.join(DATA_ROOT, data_folder)
    result_path = f'./results_3way_5shot_{result_suffix}_rev'

    print("\n" + "-"*70)
    print(f"{data_folder} (3-way {k_shot}-shot, {n_trials} trials)  ->  {result_path}")
    print("-"*70)

    # Start each dataset from the same seed for reproducible distributions
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    dataset = OurDeviceDataset(data_path, num_classes=3)

    evaluator = RandomTrialEvaluator(model, dataset, k_shot=k_shot, n_trials=n_trials)
    all_results, stats = evaluator.run_trials()

    visualizer = ResultVisualizer(save_dir=result_path)
    visualizer.save_accuracy_stats(stats)
    visualizer.plot_accuracy_distribution(stats)
    visualizer.plot_aggregated_confusion_matrix(all_results)
    representative = pick_representative_case(all_results, stats['mean_accuracy'])
    visualizer.plot_tsne(model, representative, dataset)

    return {
        'data_folder': data_folder,
        'result_suffix': result_suffix,
        'result_path': result_path,
        'k_shot': k_shot,
        'n_trials': n_trials,
        'mean_accuracy': stats['mean_accuracy'],
        'std_accuracy': stats['std_accuracy'],
        'ci95_lower': stats['ci95_lower'],
        'ci95_upper': stats['ci95_upper'],
    }


def main():
    """Run all datasets at once with the bimodal (full) model"""

    print("="*70)
    print("Random N-trial Evaluation — bimodal(full) over ALL datasets")
    print(f"  datasets={len(DATASETS)} | INPUT_CHANNELS={config.INPUT_CHANNELS}")
    print("="*70)

    # Load a single model -> reuse it across all datasets
    encoder = Encoder(
        input_channels=config.INPUT_CHANNELS,   # use full (60)
        embedding_dim=config.EMBEDDING_DIM,
        dropout=config.DROPOUT_RATE
    ).to(config.DEVICE)
    model = PrototypicalNetwork(encoder).to(config.DEVICE)

    checkpoint = torch.load(CKPT_PATH, map_location=config.DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"\nModel loaded from: {CKPT_PATH}")

    summary = []
    for data_folder, result_suffix, k_shot, n_trials in DATASETS:
        res = process_dataset(model, data_folder, result_suffix, k_shot, n_trials)
        summary.append(res)

    # Print + save the overall summary
    print("\n" + "="*70)
    print("ALL DONE — Summary (mean ± SD %, 95% CI)")
    print("="*70)
    print(f"{'suffix':<20} {'k':<3} {'N':<5} {'mean±SD':<16} {'95% CI'}")
    for r in summary:
        print(f"{r['result_suffix']:<20} {r['k_shot']:<3} {r['n_trials']:<5} "
              f"{r['mean_accuracy']:5.2f} ± {r['std_accuracy']:4.2f}    "
              f"[{r['ci95_lower']:.2f}, {r['ci95_upper']:.2f}]")

    with open('./summary_bimodal_all.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("\nSaved overall summary -> ./summary_bimodal_all.json")
    print("="*70)


if __name__ == '__main__':
    main()