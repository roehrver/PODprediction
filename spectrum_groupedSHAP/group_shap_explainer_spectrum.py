"""
Grouped SHAP Explainer for Spectral Features

The SHAP analysis with grouped features was developed entirely through iterative AI assistance 
based on my idea of grouping the spectrum by frequency bands for computational efficiency of 
the SHAP explanations. Claude Opus 4.1 was used for implementation and development.

This module implements custom SHAP value computation for grouped spectral features, where 
individual frequency bins are aggregated into meaningful frequency bands (Delta, Theta, 
Alpha, Beta, Gamma) for more interpretable and computationally efficient explanations.
"""

import numpy as np
import shap
from shap import Explanation

def explain_spectrum_groups_with_shap(
        model_predict,
        X_background,
        X_test,
        n_features=46,
        sample_background=False,
        n_samples=200
):
    """
    Compute SHAP values for spectral feature groups.

    Groups correspond to frequency bands in the spectrum.
    Features represent 1-44 Hz with 1 Hz resolution.

    Parameters:
    -----------
    model_predict : callable
        Model prediction function
    X_background : numpy.ndarray
        Background data (n_background_samples, n_features)
    X_test : numpy.ndarray
        Test samples to explain (n_test_samples, n_features)
    n_features : int
        Number of spectral features (default: 44)
    sample_background : bool
        If True, sample from background instead of using mean
    n_samples : int
        Number of samples for SHAP approximation

    Returns:
    --------
    explanation : shap.Explanation
        SHAP values for each frequency band group
    """
    # Features represent 1-44 Hz
    groups = []
    group_names = []

    # Define frequency bands and their ranges
    bands = [
        ("Delta", 1, 4),  # Features 0-2 (1-3 Hz)
        ("Theta", 4, 8),  # Features 3-6 (4-7 Hz)
        ("Lower_Alpha", 8, 12),  # Features 7-10 (8-11 Hz)
        ("Higher_Alpha", 12, 15),  # Features 11-13 (12-14 Hz)
        ("Lower_Beta", 15, 20),  # Features 14-18 (15-19 Hz)
        ("Higher_Beta", 20, 30),  # Features 19-28 (20-29 Hz)
        ("Gamma", 30, 46)  # Features 29-43 (30-44 Hz)
    ]

    # Assign features to groups based on 1 Hz resolution
    for band_name, freq_min, freq_max in bands:
        group_features = []
        for i in range(n_features):
            # Each feature i represents frequency (i+1) Hz
            freq = i + 1
            if freq_min <= freq < freq_max:
                group_features.append(i)
        if group_features:  # Only add non-empty groups
            groups.append(group_features)
            group_names.append(band_name)

    n_groups = len(groups)

    # Print group assignments for verification
    print("Frequency band assignments:")
    for i, (name, features) in enumerate(zip(group_names, groups)):
        freq_range = f"{features[0] + 1}-{features[-1] + 1} Hz"
        print(f"  {name}: features {features[0]}-{features[-1]} ({freq_range})")

    # Compute background statistics
    if not sample_background:
        feature_mask_values = np.mean(X_background, axis=0)

    # SHAP computation
    n_test = len(X_test)
    all_shap_values = []
    all_base_values = []

    for test_idx in range(n_test):
        if test_idx % 10 == 0:
            print(f"  Processing sample {test_idx}/{n_test}")

        x_test = X_test[test_idx]
        shap_values = np.zeros(n_groups)

        # Shapley value approximation
        for _ in range(n_samples):
            coalition = np.random.randint(0, 2, size=n_groups)

            for g_idx in range(n_groups):
                coalition_with = coalition.copy()
                coalition_with[g_idx] = 1
                coalition_without = coalition.copy()
                coalition_without[g_idx] = 0

                # Apply masking
                x_with = x_test.copy()
                x_without = x_test.copy()

                for g in range(n_groups):
                    if coalition_with[g] == 0:
                        if sample_background:
                            bg_idx = np.random.randint(0, len(X_background))
                            x_with[groups[g]] = X_background[bg_idx, groups[g]]
                        else:
                            x_with[groups[g]] = feature_mask_values[groups[g]]

                    if coalition_without[g] == 0:
                        if sample_background:
                            bg_idx = np.random.randint(0, len(X_background))
                            x_without[groups[g]] = X_background[bg_idx, groups[g]]
                        else:
                            x_without[groups[g]] = feature_mask_values[groups[g]]

                pred_with = model_predict(x_with.reshape(1, -1))[0]
                pred_without = model_predict(x_without.reshape(1, -1))[0]

                shap_values[g_idx] += (pred_with - pred_without) / n_samples

        # Base value
        x_masked = x_test.copy()
        for g_idx in range(n_groups):
            if sample_background:
                bg_idx = np.random.randint(0, len(X_background))
                x_masked[groups[g_idx]] = X_background[bg_idx, groups[g_idx]]
            else:
                x_masked[groups[g_idx]] = feature_mask_values[groups[g_idx]]

        base_value = model_predict(x_masked.reshape(1, -1))[0]

        all_shap_values.append(shap_values)
        all_base_values.append(base_value)

    # Create explanation
    explanation = Explanation(
        values=np.array(all_shap_values),
        base_values=np.array(all_base_values),
        data=np.ones((n_test, n_groups)),
        feature_names=group_names
    )

    return explanation
