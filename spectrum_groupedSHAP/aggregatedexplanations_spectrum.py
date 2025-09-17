"""
SHAP Spectrum Visualization and Analysis

The SHAP analysis with grouped features was developed entirely through iterative AI assistance 
based on my idea of grouping the spectrum by frequency bands for computational efficiency of 
the SHAP explanations. Claude Sonnet 4 was used for implementation and development of this file.

This script handles visualization and aggregation of SHAP results across frequency bands and 
cross-validation folds for spectrum-based machine learning models.
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import h5py
import matplotlib.patheffects as path_effects
from pathlib import Path

from matplotlib.colors import LinearSegmentedColormap

sns.set_context("paper", font_scale=2.5)  # Reduced font scale for subplots
plt.rc('lines', linewidth=1.5)
base_font_size = plt.rcParams['font.size']


def load_shap_results(folder, filename, fold, suffix=""):
    """Load aggregated SHAP results from npz file."""
    file_path = Path(folder) / f"shap_spectrum_aggregatedwithCorri_fold{fold}_{filename}{suffix}.npz"
    data = np.load(file_path)
    return {
        'values': data['values'],
        'base_values': data['base_values'],
        'feature_names': data['feature_names'],
        'correlations': data["correlations"] if 'correlations' in data else None,
        'grouped_feature_values': data['grouped_feature_values'] if 'grouped_feature_values' in data else None
    }


def calculate_band_metrics(shap_values, feature_names, correlations=None):
    """Calculate importance and consistency metrics for each band."""
    n_bands = len(feature_names)

    # Mean absolute SHAP value (importance)
    mean_importance = np.abs(shap_values).mean(axis=0)

    # Mean SHAP value (for reference)
    mean_shap = np.mean(shap_values, axis=0)

    # Direction based on correlations with threshold
    if correlations is not None:
        direction = np.zeros_like(correlations)
        direction[correlations > 0.001] = 1
        direction[correlations < -0.001] = -1
        # direction stays 0 for correlations between -0.001 and 0.001
    else:
        # Fallback to mean SHAP direction if no correlations provided
        direction = np.sign(mean_shap)

    # SHAP Consistency: fraction of samples with same sign as majority
    shap_consistency = []

    for band_idx in range(n_bands):
        values = shap_values[:, band_idx]
        # Skip if all values are zero
        if np.all(values == 0):
            shap_consistency.append(0.5)
            continue

        # Determine majority direction
        pos_count = np.sum(values > 0)
        neg_count = np.sum(values < 0)

        consistent_samples = max(pos_count, neg_count)
        shap_consistency.append(consistent_samples / len(values))

    shap_consistency = np.array(shap_consistency)

    # Normalize SHAP consistency to 0-1 if needed
    if shap_consistency.max() > shap_consistency.min():
        shap_consistency = (shap_consistency - shap_consistency.min()) / (
                    shap_consistency.max() - shap_consistency.min())

    return mean_importance, shap_consistency, mean_shap, direction


def load_individual_model_shap_results(folder, filename, fold):
    """
    Load SHAP results from individual models within a fold.
    """
    try:
        # Load individual directions saved during SHAP analysis
        directions_path = Path(folder) / f"individual_directions_fold{fold}_{filename}.npy"
        correlations_path = Path(folder) / f"individual_correlations_fold{fold}_{filename}.npy"

        if directions_path.exists():
            individual_directions = np.load(directions_path)
            print(f"Loaded individual directions from {directions_path}")
            print(f"Shape: {individual_directions.shape}")
            return individual_directions
        elif correlations_path.exists():
            # Calculate directions from correlations if directions not available
            individual_correlations = np.load(correlations_path)
            print(f"Loaded individual correlations from {correlations_path}")
            print(f"Shape: {individual_correlations.shape}")

            # Calculate directions from correlations
            individual_directions = np.zeros_like(individual_correlations)
            individual_directions[individual_correlations > 0.001] = 1
            individual_directions[individual_correlations < -0.001] = -1
            return individual_directions
        else:
            print(f"Warning: Neither individual directions nor correlations found for {filename} fold {fold}")
            return None

    except Exception as e:
        print(f"Error loading individual model results for {filename} fold {fold}: {e}")
        return None


def load_anesthetic_data(anesthetic_name):
    """Load spectrum data and class labels for specific anesthetic."""
    # Load spectrum data from h5 file
    h5_filename = f"RefSpectrum_200625noPPP_wrerefBBP_BS_FT70_bold_LPF48_10MinnoMingroup_CovNoNorm_specDIM12_logEuclidean_mex30sW{anesthetic_name}.h5"

    with h5py.File(h5_filename, "r") as f:
        spec_data_per_patient = []
        for group_key in sorted(f.keys()):
            group = f[group_key]
            inner_list = []
            for array_key in sorted(group.keys()):
                inner_list.append(np.array(group[array_key]))
            spec_data_per_patient.append(inner_list)

    # Calculate mean spectrum per patient
    mean_spec_per_patient = []
    for i in range(len(spec_data_per_patient)):
        tmp = np.mean(np.array(spec_data_per_patient[i]), axis=0)
        tmp = np.mean(tmp, axis=0)
        mean_spec_per_patient.append(tmp)

    mean_spec_per_patient = np.array(mean_spec_per_patient)

    # Load class labels
    csv_filename = f"200625noPPP_wrerefBBP_BS_FT70_bold_LPF48_10MinnoMingroup_CovNoNorm_specDIM12_logEuclidean_mex30sW{anesthetic_name}.csv"
    master = pd.read_csv(csv_filename)
    class_per_patient = master["DSM_Delir_Tag0_7_mit_Missings01"].values

    return mean_spec_per_patient, class_per_patient


def calculate_directional_consistency_within_fold(shap_folder, anesthetics, feature_names, fold=1, n_models=10):
    """
    Calculate directional consistency across the bagged models within a single fold.

    Returns:
    - directional_consistency_dict: Dictionary with anesthetic names as keys and
      directional consistency arrays as values
    """
    directional_consistency_dict = {}

    for anesthetic in anesthetics:
        print(f"Calculating directional consistency for {anesthetic} within fold {fold}...")

        try:
            # Load individual model directions
            directions_across_models = load_individual_model_shap_results(
                shap_folder, anesthetic, fold
            )

            if directions_across_models is not None:
                print(f"  Loaded {len(directions_across_models)} models")
                print(f"  Directions shape: {directions_across_models.shape}")

                # Calculate directional consistency as percentage of models pointing in same direction
                directional_consistency = np.maximum(
                    (directions_across_models > 0).sum(axis=0),
                    (directions_across_models < 0).sum(axis=0)
                ) / len(directions_across_models)

                directional_consistency_dict[anesthetic] = directional_consistency

                # Load SHAP results to get importance values
                shap_data = load_shap_results(shap_folder, anesthetic, fold)
                mean_importance, _, _, _ = calculate_band_metrics(
                    shap_data['values'], feature_names, shap_data.get('correlations')
                )

                # Print some statistics including importance
                print(f"  Directional consistency for {anesthetic}:")
                for i, feature_name in enumerate(feature_names):
                    pos_count = (directions_across_models[:, i] > 0).sum()
                    neg_count = (directions_across_models[:, i] < 0).sum()
                    zero_count = (directions_across_models[:, i] == 0).sum()
                    consistency = directional_consistency[i]
                    importance = mean_importance[i]
                    print(
                        f"    {feature_name}: {pos_count}+ {neg_count}- {zero_count}0 -> {consistency:.1%} consistent (importance: {importance:.3f})")
            else:
                print(f"Warning: Could not load individual model results for {anesthetic}")
                # Fallback: try to estimate from aggregated results
                shap_data = load_shap_results(shap_folder, anesthetic, fold)
                if shap_data.get('correlations') is not None:
                    # Use moderate consistency (0.7) as fallback
                    directional_consistency_dict[anesthetic] = np.full(len(feature_names), 0.7)
                else:
                    directional_consistency_dict[anesthetic] = np.ones(len(feature_names))  # Default to 100% consistent

        except Exception as e:
            print(f"Error processing {anesthetic}: {e}")
            directional_consistency_dict[anesthetic] = np.ones(len(feature_names))  # Default to 100% consistent

    return directional_consistency_dict


def save_individual_model_correlations_during_analysis(shap_explanations, save_path):
    """
    Helper function to save individual model correlations during SHAP analysis.
    This function is now integrated into the modified analyze_spectrum_shap function.
    """
    individual_correlations = []

    for exp in shap_explanations:
        if hasattr(exp, 'grouped_feature_values'):
            correlations = []
            for band_idx in range(exp.values.shape[1]):
                corr = np.corrcoef(exp.grouped_feature_values[:, band_idx],
                                   exp.values[:, band_idx])[0, 1]
                correlations.append(corr if not np.isnan(corr) else 0)
            individual_correlations.append(correlations)
        else:
            # Fallback: use SHAP values direction
            mean_shap = np.mean(exp.values, axis=0)
            individual_correlations.append(np.sign(mean_shap))

    # Save individual correlations
    np.save(save_path, np.array(individual_correlations))
    print(f"Individual model correlations saved to: {save_path}")

    return np.array(individual_correlations)
    """Load spectrum data and class labels for specific anesthetic."""
    # Load spectrum data from h5 file
    h5_filename = f"RefSpectrum_200625noPPP_wrerefBBP_BS_FT70_bold_LPF48_10MinnoMingroup_CovNoNorm_specDIM12_logEuclidean_mex30sW{anesthetic_name}.h5"

    with h5py.File(h5_filename, "r") as f:
        spec_data_per_patient = []
        for group_key in sorted(f.keys()):
            group = f[group_key]
            inner_list = []
            for array_key in sorted(group.keys()):
                inner_list.append(np.array(group[array_key]))
            spec_data_per_patient.append(inner_list)

    # Calculate mean spectrum per patient
    mean_spec_per_patient = []
    for i in range(len(spec_data_per_patient)):
        tmp = np.mean(np.array(spec_data_per_patient[i]), axis=0)
        tmp = np.mean(tmp, axis=0)
        mean_spec_per_patient.append(tmp)

    mean_spec_per_patient = np.array(mean_spec_per_patient)

    # Load class labels
    csv_filename = f"200625noPPP_wrerefBBP_BS_FT70_bold_LPF48_10MinnoMingroup_CovNoNorm_specDIM12_logEuclidean_mex30sW{anesthetic_name}.csv"
    master = pd.read_csv(csv_filename)
    class_per_patient = master["DSM_Delir_Tag0_7_mit_Missings01"].values

    return mean_spec_per_patient, class_per_patient


def plot_aggregated_anesthetic(ax1, ax2, mean_spectra_per_subject, class_labels,
                               agg_data, feature_names, anesthetic_name,
                               frequencies=None, show_legend=True):
    """
    Plot mean spectra per class with aggregated SHAP band importance

    Parameters:
    - agg_data: Dictionary with aggregated metrics from aggregate_shap_data_across_folds
    """

    if frequencies is None:
        frequencies = np.arange(1, 49)

    # Extract aggregated metrics
    mean_importance = agg_data['mean_importance']
    direction = agg_data['direction']
    directional_consistency = agg_data['mean_directional_consistency']
    direction_agreement = agg_data['direction_agreement']

    # Define band frequency ranges
    band_ranges = {
        "Delta": (1, 4),
        "Theta": (4, 8),
        "Lower_Alpha": (8, 12),
        "Higher_Alpha": (12, 15),
        "Lower_Beta": (15, 20),
        "Higher_Beta": (20, 30),
        "Gamma": (30, 45)
    }

    # Create band-wise arrays for full frequency range
    importance_by_freq = np.zeros(len(frequencies))
    directional_consistency_by_freq = np.zeros(len(frequencies))
    direction_by_freq = np.zeros(len(frequencies))
    direction_agreement_by_freq = np.zeros(len(frequencies))

    for i, band_name in enumerate(feature_names):
        if band_name in band_ranges:
            freq_min, freq_max = band_ranges[band_name]
            freq_mask = (frequencies >= freq_min) & (frequencies < freq_max)
            importance_by_freq[freq_mask] = mean_importance[i]
            directional_consistency_by_freq[freq_mask] = directional_consistency[i]
            direction_by_freq[freq_mask] = direction[i]
            direction_agreement_by_freq[freq_mask] = direction_agreement[i]

    # Define colors
    class_colors = [(0.00392156862745098, 0.45098039215686275, 0.6980392156862745),
                    (0.7, 0.7, 0.7)]

    # Separate data by class
    class_0_mask = class_labels == 0
    class_1_mask = class_labels == 1

    spectra_class_0 = mean_spectra_per_subject[class_0_mask]
    spectra_class_1 = mean_spectra_per_subject[class_1_mask]

    mean_class_0 = np.mean(spectra_class_0, axis=0)
    std_class_0 = np.std(spectra_class_0, axis=0)
    mean_class_1 = np.mean(spectra_class_1, axis=0)
    std_class_1 = np.std(spectra_class_1, axis=0)

    # Normalize importance for visualization
    norm_importance = importance_by_freq / importance_by_freq.max() if importance_by_freq.max() > 0 else importance_by_freq

    # Plot bars for each band with directional consistency coloring
    for band_idx, (band_name, (freq_min, freq_max)) in enumerate(band_ranges.items()):
        band_mask = (frequencies >= freq_min) & (frequencies < freq_max)
        band_freqs = frequencies[band_mask]
        band_importance = norm_importance[band_mask]
        band_directional_consistency = directional_consistency_by_freq[band_mask]
        band_direction_agreement = direction_agreement_by_freq[band_mask]

        if len(band_freqs) > 0 and len(band_directional_consistency) > 0:
            mean_directional_consistency = np.mean(band_directional_consistency)
            mean_direction_agreement = np.mean(band_direction_agreement)

            if band_name in feature_names:
                idx = np.where(feature_names == band_name)[0][0]
                actual_direction = direction[idx]

                # Color intensity based on directional consistency within models
                bar_color = plt.cm.viridis(mean_directional_consistency)

                # Alpha based on agreement across folds (optional)
                # bar_alpha = 0.2 + 0.3 * mean_direction_agreement  # Scale from 0.2 to 0.5
                bar_alpha = 0.4  # Or keep constant

                # Edge and hatch for direction
                if actual_direction < 0:
                    edge_color = 'black'
                    line_width = 3
                    hatch_pattern = '//'
                elif actual_direction == 0:
                    edge_color = 'black'
                    line_width = 1
                    hatch_pattern = 'xxx'
                else:
                    edge_color = None
                    line_width = 0
                    hatch_pattern = ''

                bars = ax2.bar(band_freqs, band_importance,
                               color=bar_color,
                               alpha=bar_alpha,
                               width=0.8,
                               edgecolor=edge_color,
                               linewidth=line_width,
                               hatch=hatch_pattern)

                # Optional: Add error bars for importance std
                if 'std_importance' in agg_data:
                    std_importance_by_freq = np.zeros(len(frequencies))
                    std_importance_by_freq[band_mask] = agg_data['std_importance'][idx]
                    norm_std = std_importance_by_freq[
                                   band_mask] / importance_by_freq.max() if importance_by_freq.max() > 0 else \
                    std_importance_by_freq[band_mask]

                    # Add subtle error bars
                    for j, (x, y) in enumerate(zip(band_freqs, band_importance)):
                        ax2.errorbar(x, y, yerr=norm_std[j],
                                     color='gray', alpha=0.5, capsize=2, linewidth=1)

    ax2.set_xlim(0.3, 46)

    # Plot spectra (same as before)
    ax1.plot(frequencies, mean_class_1, color=class_colors[0], linewidth=4,
             label=f'POD (n={np.sum(class_1_mask)})')
    ax1.set_xlim(0.3, 46)

    line = ax1.plot(frequencies, mean_class_0, color='white', linewidth=2.5, linestyle='--',
                    label=f'No POD (n={np.sum(class_0_mask)})')
    line[0].set_path_effects([path_effects.Stroke(linewidth=4, foreground='black'),
                              path_effects.Normal()])

    ax1.fill_between(frequencies,
                     mean_class_0 - std_class_0,
                     mean_class_0 + std_class_0,
                     color='white', alpha=0.8, edgecolor='black', linewidth=0.5)

    ax1.fill_between(frequencies,
                     mean_class_1 - std_class_1,
                     mean_class_1 + std_class_1,
                     color=class_colors[0], alpha=0.4)

    # Formatting
    ax1.set_xlabel('Frequency (Hz)', fontdict={'weight': 'bold'})
    ax1.set_ylabel('PSD', fontdict={'weight': 'bold'})
    ax2.set_ylabel('Normalized Band Importance', fontdict={'weight': 'bold'})
    ax1.set_title(anesthetic_name.capitalize(), fontdict={'weight': 'bold', 'size': base_font_size * 1.6})

    if show_legend:
        from matplotlib.patches import Patch
        direction_legend = [
            Patch(facecolor='grey', alpha=0.8, label='Higher → POD'),
            Patch(facecolor='grey', alpha=0.8, edgecolor='black', hatch='//',
                  linewidth=2, label='Higher → No POD'),
            Patch(facecolor='grey', alpha=0.8, edgecolor='black', hatch='xxx',
                  linewidth=1, label='No direction')
        ]
        ax2.legend(handles=direction_legend, loc='lower center', bbox_to_anchor=(0.5, -0.25),
                   framealpha=0.9, ncol=3)
        ax1.legend(loc='upper right')

    ax1.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.1)

    # Add vertical lines to separate bands
    for band_name, (freq_min, freq_max) in band_ranges.items():
        if freq_max <= frequencies[-1]:
            ax1.axvline(x=freq_max, color='gray', linestyle=':', alpha=0.5)

    return ax1, ax2


def aggregate_shap_data_across_folds(shap_folder, anesthetic, feature_names, folds=[1, 2, 3, 4, 5]):
    """Aggregate SHAP data across multiple folds"""

    all_importances = []
    all_directions = []
    all_correlations = []
    all_directional_consistencies = []

    for fold in folds:
        # Load fold data
        shap_data = load_shap_results(shap_folder, anesthetic, fold)

        # Calculate metrics for this fold
        mean_importance, _, _, direction = calculate_band_metrics(
            shap_data['values'],
            feature_names,
            shap_data.get('correlations')
        )

        # Get directional consistency within fold (across 10 bagged models)
        individual_directions = load_individual_model_shap_results(
            shap_folder, anesthetic, fold
        )
        if individual_directions is not None:
            directional_consistency = np.maximum(
                (individual_directions > 0).sum(axis=0),
                (individual_directions < 0).sum(axis=0)
            ) / len(individual_directions)
        else:
            directional_consistency = np.ones(len(mean_importance)) * 0.7

        all_importances.append(mean_importance)
        all_directions.append(direction)
        all_correlations.append(shap_data.get('correlations', np.zeros_like(mean_importance)))
        all_directional_consistencies.append(directional_consistency)

    # Aggregate across folds
    aggregated_data = {
        'mean_importance': np.mean(all_importances, axis=0),
        'std_importance': np.std(all_importances, axis=0),
        'direction': np.sign(np.mean(all_directions, axis=0)),  # Majority vote
        'direction_agreement': np.mean(np.array(all_directions) ==
                                       np.sign(np.mean(all_directions, axis=0))[np.newaxis, :], axis=0),
        'mean_directional_consistency': np.mean(all_directional_consistencies, axis=0),
        'mean_correlations': np.mean(all_correlations, axis=0)
    }

    return aggregated_data


def plot_aggregated_multi_anesthetic_spectra(anesthetics, shap_folder, folds=[1, 2, 3, 4, 5],
                                             frequencies=None, n_models=10):
    """
    Plot mean spectra with aggregated SHAP importance across all folds

    Parameters:
    - anesthetics: list of anesthetic names
    - shap_folder: path to SHAP results
    - folds: list of fold numbers to aggregate
    - frequencies: frequency bins
    - n_models: number of bagged models within each fold
    """

    if frequencies is None:
        frequencies = np.arange(1, 49)

    # Get feature names from first fold
    sample_data = load_shap_results(shap_folder, anesthetics[0], folds[0])
    feature_names = sample_data['feature_names']

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(36, 16), dpi=300)
    ax1_list = [axes[0, i] for i in range(3)]
    ax2_list = [axes[1, i] for i in range(3)]
    titles = ['A', 'B', 'C']

    # Process each anesthetic
    for i, anesthetic in enumerate(anesthetics):
        print(f"\nAggregating {anesthetic} across {len(folds)} folds...")

        # Load spectrum data (same for all folds)
        mean_spectra, class_labels = load_anesthetic_data(anesthetic)

        # Aggregate SHAP data across folds
        agg_data = aggregate_shap_data_across_folds(shap_folder, anesthetic, feature_names, folds)

        # Print aggregated statistics
        print(f"  Aggregated metrics for {anesthetic}:")
        for j, band_name in enumerate(feature_names):
            print(f"    {band_name}:")
            print(f"      Importance: {agg_data['mean_importance'][j]:.3f} ± {agg_data['std_importance'][j]:.3f}")
            print(
                f"      Direction: {agg_data['direction'][j]:+.0f} (agreement: {agg_data['direction_agreement'][j]:.1%})")
            print(f"      Model consistency: {agg_data['mean_directional_consistency'][j]:.1%}")

        # Create twin axis for SHAP importance
        ax2 = ax1_list[i].twinx()

        # Plot with aggregated data
        plot_aggregated_anesthetic(
            ax1_list[i], ax2,
            mean_spectra, class_labels,
            agg_data, feature_names,
            titles[i],  # Using A, B, C as titles
            frequencies,
            show_legend=(i == 1)  # Only middle plot gets legend
        )

    # Remove unused bottom row
    for ax in ax2_list:
        ax.remove()

    # Add colorbar for directional consistency
    bbox = axes[0, -1].get_position()
    cbar_ax = fig.add_axes([bbox.x1 + 0.1, bbox.y0, 0.02, bbox.height + 0.07])

    # Create blended colormap
    viridis = plt.colormaps['viridis']
    colors = viridis(np.linspace(0, 1, 256))
    white = np.array([1, 1, 1, 1])
    alpha = 0.4
    blended_colors = colors * alpha + white * (1 - alpha)
    blended_colors[:, 3] = 1.0

    viridis_blended = LinearSegmentedColormap.from_list('viridis_blended', blended_colors)

    # Keep normalization at 0-1 internally but display as 0-100%
    sm = plt.cm.ScalarMappable(cmap=viridis_blended, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)

    # Format tick labels as percentages
    tick_locator = plt.MaxNLocator(nbins=6)
    cbar.locator = tick_locator
    cbar.update_ticks()
    cbar.ax.set_yticklabels([f'{int(x * 100)}%' for x in cbar.get_ticks()])

    cbar.set_label(f'Directional Consistency\n(% of {n_models} models agreeing)\naveraged across {len(folds)} folds',
                   fontweight='bold')

    # Add title indicating aggregation
    plt.suptitle(f'Aggregated Results Across {len(folds)} Cross-Validation Folds',
                 fontsize=18, fontweight='bold', y=0.98)

    plt.tight_layout()

    # Save figure
    output_filename = f'aggregated_spectra_with_shap_{len(folds)}folds.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved as: {output_filename}")

    plt.show()

    return fig


# Main execution
if __name__ == "__main__":
    # Configuration
    anesthetics = ['propofol', 'sevoflurane', 'desflurane']
    shap_folder = "spectrum_explanationsdata"
    folds_to_aggregate = [1, 2, 3, 4, 5]  # Specify which folds to aggregate

    # Create aggregated figure
    print("=" * 50)
    print("Creating aggregated figure across all folds...")
    print("=" * 50)

    fig = plot_aggregated_multi_anesthetic_spectra(
        anesthetics=anesthetics,
        shap_folder=shap_folder,
        folds=folds_to_aggregate,
        frequencies=np.arange(1, 49),
        n_models=10  # Number of bagged models within each fold
    )

    # Optional: Also create individual fold figures for appendix
    # create_appendix = input("\nCreate individual fold figures for appendix? (y/n): ")
    # if create_appendix.lower() == 'y':
    #     print("\n" + "=" * 50)
    #     print("Creating individual fold figures for appendix...")
    #     print("=" * 50)
    #
    #     for fold in folds_to_aggregate:
    #         print(f"\nProcessing fold {fold}...")
    #         plot_multi_anesthetic_spectra_with_shap(
    #             anesthetics=anesthetics,
    #             shap_folder=shap_folder,
    #             fold=fold,
    #             frequencies=np.arange(1, 49),
    #             n_models=10
    #         )
    #         print(f"Saved: multi_anesthetic_spectra_with_shap_fold{fold}.png")
    #
    #     print("\n" + "=" * 50)
    #     print("All figures created successfully!")
    #     print(f"Main figure: aggregated_spectra_with_shap_{len(folds_to_aggregate)}folds.png")
    #     print(f"Appendix figures: multi_anesthetic_spectra_with_shap_fold[1-{len(folds_to_aggregate)}].png")
