"""
Spectrum SHAP Analysis Pipeline

The SHAP analysis with grouped features was developed entirely through iterative AI assistance 
based on my idea of grouping the spectrum by frequency bands for computational efficiency of 
the SHAP explanations. Claude Opus 4.1 was used for implementation and development.

This script provides the complete pipeline for loading models, computing SHAP values with 
grouped spectral features, and analyzing directional consistency across bagged models and 
cross-validation folds.
"""

import numpy as np
import joblib
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from group_shap_explainer import explain_spectrum_groups_with_shap

def load_spectrum_shap_data(folder, filename, overallfold, model_folder):
    """Load saved data for spectrum SHAP analysis with new structure."""
    shap_folder = Path(folder) / "shap_data"

    # Load metadata
    with open(shap_folder / f"metadata_spectrum_fold{overallfold}_{filename}.json", 'r') as f:
        metadata = json.load(f)

    n_models = metadata['n_models']

    # Load fixed training data (same for all models)
    fixed_train = np.load(shap_folder / f"fixed_train_spectrum_fold{overallfold}_{filename}.npy")
    fixed_train_labels = np.load(shap_folder / f"fixed_train_labels_spectrum_fold{overallfold}_{filename}.npy")

    # Load models and test data
    models = []
    test_list = []
    test_labels_list = []

    for i in range(n_models):
        # Load model
        model_path = Path(model_folder) / f"pipelineSpectrum{overallfold}{filename}{i}.pkl"
        model_data = joblib.load(model_path)
        models.append(model_data[0])

        # Load test data for this bag
        test_data = np.load(shap_folder / f"test_samples_spectrum_fold{overallfold}_{filename}_bag{i}.npy")
        test_labels = np.load(shap_folder / f"test_labels_spectrum_fold{overallfold}_{filename}_bag{i}.npy")
        test_list.append(test_data)
        test_labels_list.append(test_labels)

    return models, fixed_train, test_list, fixed_train_labels, test_labels_list, metadata


def compute_spectrum_shap(models, fixed_train, test_list, fixed_train_labels, test_labels_list,
                          n_background_samples=500, n_test_samples=100):
    """Compute SHAP values for spectrum models using fixed background."""

    shap_explanations = []

    # Use the same background for all models
    if n_background_samples < len(fixed_train):
        # Create balanced background from fixed training set
        pos_idx = np.where(fixed_train_labels > 0.5)[0]
        neg_idx = np.where(fixed_train_labels < 0.5)[0]

        n_per_class = n_background_samples // 2
        sampled_pos = np.random.choice(pos_idx, min(n_per_class, len(pos_idx)), replace=False)
        sampled_neg = np.random.choice(neg_idx, min(n_per_class, len(neg_idx)), replace=False)

        background_idx = np.concatenate([sampled_pos, sampled_neg])
        background_data = fixed_train[background_idx]
    else:
        background_data = fixed_train

    # Band definitions
    bands = [
        ("Delta", 1, 4),
        ("Theta", 4, 8),
        ("Lower_Alpha", 8, 12),
        ("Higher_Alpha", 12, 15),
        ("Lower_Beta", 15, 20),
        ("Higher_Beta", 20, 30),
        ("Gamma", 30, 46)
    ]

    # Create groups
    groups = []
    group_names = []
    for band_name, freq_min, freq_max in bands:
        group_features = []
        for i in range(fixed_train.shape[1]):  # n_features
            freq = i + 1
            if freq_min <= freq < freq_max:
                group_features.append(i)
        if group_features:
            groups.append(group_features)
            group_names.append(band_name)

    for i, (model, test) in enumerate(zip(models, test_list)):
        print(f"Computing SHAP for model {i + 1}/{len(models)}")

        # Test data is already sampled per bag
        # Further sample if needed
        if n_test_samples < len(test):
            test_idx = np.random.choice(len(test), n_test_samples, replace=False)
            test_data = test[test_idx]
        else:
            test_data = test

        # Calculate grouped feature values by averaging within each band
        grouped_test_features = np.zeros((len(test_data), len(groups)))
        for g_idx, group_features in enumerate(groups):
            grouped_test_features[:, g_idx] = np.mean(test_data[:, group_features], axis=1)

        # Define prediction function
        def model_predict(x):
            return model.predict_proba(x)[:, 1]

        # Compute SHAP
        explanation = explain_spectrum_groups_with_shap(
            model_predict,
            background_data,
            test_data,
            n_features=fixed_train.shape[1],
            sample_background=False,  # Already sampled
            n_samples=1000
        )

        # Add the actual feature values to the explanation
        explanation.grouped_feature_values = grouped_test_features
        shap_explanations.append(explanation)

    return shap_explanations


def calculate_individual_model_correlations_and_directions(shap_explanations):
    """
    Calculate correlations and directions for each individual model before aggregation.

    Returns:
    - individual_correlations: array of shape (n_models, n_features)
    - individual_directions: array of shape (n_models, n_features)
    """
    individual_correlations = []
    individual_directions = []

    for exp in shap_explanations:
        correlations = []

        if hasattr(exp, 'grouped_feature_values') and exp.grouped_feature_values is not None:
            # Calculate correlations between feature values and SHAP values
            for band_idx in range(exp.values.shape[1]):
                corr = np.corrcoef(exp.grouped_feature_values[:, band_idx],
                                   exp.values[:, band_idx])[0, 1]
                correlations.append(corr if not np.isnan(corr) else 0)
        else:
            # Fallback: use mean SHAP values as proxy for correlation
            mean_shap = np.mean(exp.values, axis=0)
            correlations = mean_shap.tolist()

        correlations = np.array(correlations)
        individual_correlations.append(correlations)

        # Calculate direction based on correlations with threshold
        direction = np.zeros_like(correlations)
        direction[correlations > 0.001] = 1
        direction[correlations < -0.001] = -1
        # direction stays 0 for correlations between -0.001 and 0.001

        individual_directions.append(direction)

    return np.array(individual_correlations), np.array(individual_directions)


def aggregate_shap_explanations(shap_explanations):
    """Aggregate SHAP values across multiple models."""
    n_samples = shap_explanations[0].values.shape[0]
    n_groups = shap_explanations[0].values.shape[1]
    n_models = len(shap_explanations)

    # Aggregate
    aggregated_values = np.zeros((n_samples, n_groups))
    aggregated_base = np.zeros(n_samples)

    for exp in shap_explanations:
        aggregated_values += exp.values
        aggregated_base += exp.base_values

    aggregated_values /= n_models
    aggregated_base /= n_models

    # Handle grouped feature values if present
    if hasattr(shap_explanations[0], 'grouped_feature_values'):
        grouped_feature_values = shap_explanations[0].grouped_feature_values
        print(f"Grouped feature values shape: {grouped_feature_values.shape}")

        correlations = []
        for band_idx in range(aggregated_values.shape[1]):
            corr = np.corrcoef(grouped_feature_values[:, band_idx],
                               aggregated_values[:, band_idx])[0, 1]
            correlations.append(corr if not np.isnan(corr) else 0)
        print(f"Aggregated correlations: {correlations}")
    else:
        grouped_feature_values = None
        correlations = None

    return {
        'values': aggregated_values,
        'base_values': aggregated_base,
        'feature_names': shap_explanations[0].feature_names,
        'correlations': np.array(correlations) if correlations else None,
        'grouped_feature_values': grouped_feature_values
    }


def plot_shap_summary(aggregated_explanation, save_path=None, title="SHAP Summary"):
    """Create comprehensive SHAP summary plots."""

    values = aggregated_explanation['values']
    feature_names = aggregated_explanation['feature_names']

    # Calculate statistics
    mean_abs_shap = np.abs(values).mean(axis=0)
    sorted_idx = np.argsort(mean_abs_shap)[::-1]

    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))

    # 1. Bar plot of mean absolute SHAP values
    ax1 = plt.subplot(2, 2, 1)
    ax1.barh(range(len(sorted_idx)), mean_abs_shap[sorted_idx])
    ax1.set_yticks(range(len(sorted_idx)))
    ax1.set_yticklabels([feature_names[i] for i in sorted_idx])
    ax1.set_xlabel('Mean |SHAP value|')
    ax1.set_title('Feature Importance')

    # 2. Box plot of SHAP value distributions
    ax2 = plt.subplot(2, 2, 2)
    shap_data = [values[:, i] for i in sorted_idx]
    ax2.boxplot(shap_data, vert=False, labels=[feature_names[i] for i in sorted_idx])
    ax2.set_xlabel('SHAP value')
    ax2.set_title('SHAP Value Distribution')

    # 3. Violin plot for top features
    ax3 = plt.subplot(2, 2, 3)
    top_n = min(5, len(feature_names))
    top_features = sorted_idx[:top_n]
    data_for_violin = []
    labels_for_violin = []
    for i in top_features:
        data_for_violin.extend(values[:, i])
        labels_for_violin.extend([feature_names[i]] * len(values[:, i]))

    import pandas as pd
    df_violin = pd.DataFrame({'SHAP value': data_for_violin, 'Feature': labels_for_violin})
    sns.violinplot(data=df_violin, x='SHAP value', y='Feature', ax=ax3)
    ax3.set_title(f'Top {top_n} Features - Distribution')

    # 4. Heatmap of SHAP values for first 20 samples
    ax4 = plt.subplot(2, 2, 4)
    n_samples_show = min(20, values.shape[0])
    im = ax4.imshow(values[:n_samples_show, sorted_idx].T, aspect='auto', cmap='RdBu_r')
    ax4.set_yticks(range(len(sorted_idx)))
    ax4.set_yticklabels([feature_names[i] for i in sorted_idx])
    ax4.set_xlabel('Sample')
    ax4.set_title('SHAP Values Heatmap')
    plt.colorbar(im, ax=ax4)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    plt.show()

    # Print summary statistics
    print("\nSHAP Summary Statistics:")
    print("-" * 50)
    for i in sorted_idx:
        print(f"{feature_names[i]:20s}: mean |SHAP| = {mean_abs_shap[i]:.4f}, "
              f"std = {np.std(values[:, i]):.4f}")


def analyze_spectrum_shap(folder, filename, overallfold, model_folder,
                          n_background_samples=5000, n_test_samples=1000):
    """Complete SHAP analysis for spectrum models with new data structure."""

    print("Loading spectrum data...")
    models, fixed_train, test_list, fixed_train_labels, test_labels_list, metadata = load_spectrum_shap_data(
        folder, filename, overallfold, model_folder
    )

    print(f"Loaded {len(models)} models")
    print(f"Features: {metadata['n_features']}")
    print(f"Fixed training samples: {metadata.get('n_fixed_train_samples', 'N/A')}")
    print(f"Fixed test samples per bag: {metadata.get('n_fixed_test_samples', 'N/A')}")

    print("\nComputing SHAP values...")
    shap_explanations = compute_spectrum_shap(
        models, fixed_train, test_list, fixed_train_labels, test_labels_list,
        n_background_samples, n_test_samples
    )

    print("\nCalculating individual model correlations and directions...")
    individual_correlations, individual_directions = calculate_individual_model_correlations_and_directions(
        shap_explanations
    )

    print(f"Individual correlations shape: {individual_correlations.shape}")
    print(f"Individual directions shape: {individual_directions.shape}")

    # Print directional consistency statistics
    feature_names = shap_explanations[0].feature_names
    print("\nDirectional Consistency across models:")
    print("-" * 50)
    for i, feature_name in enumerate(feature_names):
        pos_count = (individual_directions[:, i] > 0).sum()
        neg_count = (individual_directions[:, i] < 0).sum()
        zero_count = (individual_directions[:, i] == 0).sum()
        consistency = max(pos_count, neg_count) / len(individual_directions)
        print(f"{feature_name:20s}: {pos_count:2d}+ {neg_count:2d}- {zero_count:2d}0 -> {consistency:.1%} consistent")

    # Save individual model results
    individual_correlations_path = Path(folder) / f"individual_correlations_fold{overallfold}_{filename}.npy"
    individual_directions_path = Path(folder) / f"individual_directions_fold{overallfold}_{filename}.npy"

    np.save(individual_correlations_path, individual_correlations)
    np.save(individual_directions_path, individual_directions)

    print(f"Individual correlations saved to: {individual_correlations_path}")
    print(f"Individual directions saved to: {individual_directions_path}")

    print("\nAggregating SHAP values...")
    aggregated = aggregate_shap_explanations(shap_explanations)

    # Save aggregated results
    save_path = Path(folder) / f"shap_spectrum_aggregatedwithCorri_fold{overallfold}_{filename}.npz"
    np.savez(save_path, **aggregated)
    print(f"Aggregated SHAP values saved to: {save_path}")

    # Create plots
    plot_path = Path(folder) / f"shap_spectrum_summary_fold{overallfold}_{filename}.png"
    plot_shap_summary(aggregated, save_path=plot_path, title="Spectrum SHAP Analysis")

    return aggregated, individual_correlations, individual_directions


if __name__ == "__main__":
    # Example usage for spectrum
    folder = "spectrum_explanationsdata"
    filename = "desflurane"
    print(f"Analyzing spectrum for {filename}")
    overallfold = 4
    model_folder = f"spectrum_models{filename}"

    aggregated, individual_correlations, individual_directions = analyze_spectrum_shap(
        folder, filename, overallfold, model_folder,
        n_background_samples=5100, n_test_samples=1100
    )
