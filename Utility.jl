"""
Utilities module for risk evaluation system.
Provides helper functions for performance metrics.
It was written by Claude Opus 4.1.
"""
module Utilities

export calculate_performance_metrics,
         print_performance_summary,
         validate_data_consistency

using Statistics
using PyCall
using JSON
using DataFrames
using CSV

# Import sklearn metrics
const sklearn_metrics = PyCall.PyNULL()

function __init__()
    copy!(sklearn_metrics, pyimport("sklearn.metrics"))
end

"""
    calculate_performance_metrics(true_labels, predictions; threshold=0.5)

Calculate comprehensive performance metrics for binary classification.

# Arguments
- `true_labels`: Ground truth labels (0 or 1)
- `predictions`: Predicted probabilities or scores
- `threshold`: Decision threshold (default: 0.5)

# Returns
- Dictionary with metrics (AUC, accuracy, sensitivity, specificity, etc.)
"""
function calculate_performance_metrics(
    true_labels::Vector,
    predictions::Vector;
    threshold::Float64 = 0.5
)
    # Convert predictions to binary
    binary_predictions = [p >= threshold ? 1 : 0 for p in predictions]
    
    # Calculate metrics
    metrics = Dict{String, Float64}()
    
    # ROC AUC
    metrics["roc_auc"] = sklearn_metrics.roc_auc_score(true_labels, predictions)
    
    # Accuracy metrics
    metrics["accuracy"] = sklearn_metrics.accuracy_score(true_labels, binary_predictions)
    metrics["balanced_accuracy"] = sklearn_metrics.balanced_accuracy_score(
        true_labels, binary_predictions
    )
    
    # Precision, Recall, F1
    metrics["precision"] = sklearn_metrics.precision_score(
        true_labels, binary_predictions, zero_division=0
    )
    metrics["recall"] = sklearn_metrics.recall_score(
        true_labels, binary_predictions, zero_division=0
    )
    metrics["f1_score"] = sklearn_metrics.f1_score(
        true_labels, binary_predictions, zero_division=0
    )
    
    # Confusion matrix values
    tn, fp, fn, tp = sklearn_metrics.confusion_matrix(
        true_labels, binary_predictions
    )
    
    metrics["true_positives"] = Float64(tp)
    metrics["true_negatives"] = Float64(tn)
    metrics["false_positives"] = Float64(fp)
    metrics["false_negatives"] = Float64(fn)
    
    # Sensitivity and Specificity
    metrics["sensitivity"] = tp / (tp + fn)  # Same as recall
    metrics["specificity"] = tn / (tn + fp)
    
    # Matthews Correlation Coefficient
    metrics["mcc"] = sklearn_metrics.matthews_corrcoef(true_labels, binary_predictions)
    
    return metrics
end

"""
    print_performance_summary(metrics::Dict; model_name="Model")

Pretty print performance metrics.

# Arguments
- `metrics`: Dictionary of metrics
- `model_name`: Name to display
"""
function print_performance_summary(metrics::Dict; model_name::String = "Model")
    println("\n" * "="^50)
    println("Performance Summary: $model_name")
    println("="^50)
    
    # Key metrics to highlight
    key_metrics = ["roc_auc", "balanced_accuracy", "sensitivity", "specificity", "f1_score"]
    
    println("\nKey Metrics:")
    println("-"^30)
    for metric in key_metrics
        if haskey(metrics, metric)
            value = metrics[metric]
            println("  $(rpad(metric, 20)): $(round(value, digits=3))")
        end
    end
    
    println("\nConfusion Matrix:")
    println("-"^30)
    if all(haskey(metrics, k) for k in ["true_positives", "true_negatives", 
                                         "false_positives", "false_negatives"])
        tp = Int(metrics["true_positives"])
        tn = Int(metrics["true_negatives"])
        fp = Int(metrics["false_positives"])
        fn = Int(metrics["false_negatives"])
        
        println("  Predicted")
        println("     0    1")
        println("  0  $tn   $fp")
        println("  1  $fn   $tp")
    end
    
    println("="^50)
end

end