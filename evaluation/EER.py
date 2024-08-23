import numpy as np

def compute_det_curve(target_scores, nontarget_scores):
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))  # false rejection rates
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))  # false acceptance rates
    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))  # Thresholds are the sorted scores

    return frr, far, thresholds

def compute_eer(target_scores, nontarget_scores):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]

def evaluate_eer(logits, true_labels):
    """
    Evaluate EER (Equal Error Rate) given model logits and true labels.

    Args:
        logits (np.ndarray): Model's predicted logits, should be of shape (num_samples, num_classes) for binary classification.
        true_labels (np.ndarray): True binary labels (1 for genuine, 0 for imposters/spoofs).

    Returns:
        eer (float): Equal Error Rate (EER) score.
        threshold (float): The threshold at which EER occurs.
    """
    logits = np.array(logits)
    true_labels = np.array(true_labels)

    if logits.ndim > 1:
        # Assuming binary classification, extract logits for one of the classes (e.g., positive class)
        logits = logits[:, 1]  # Select the logits corresponding to the positive class

    # Separate target and nontarget scores based on true labels
    target_scores = logits[true_labels == 1]
    nontarget_scores = logits[true_labels == 0]

    eer, threshold = compute_eer(target_scores, nontarget_scores)
    return eer, threshold




# len(logits[0])


# Calculate EER using the logits and true labels
eer, threshold = evaluate_eer(logits, y_true)
print(f"EER: {eer:.4f}, Threshold: {threshold:.4f}")