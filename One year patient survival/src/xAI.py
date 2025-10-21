"""explainable_ai.py
Memory-optimized SHAP and LIME helpers. These functions take a trained model + data and
write plots/artifacts to disk under an output directory.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc
import psutil

# try imports
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

try:
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except Exception:
    LIME_AVAILABLE = False


def get_memory_usage_mb():
    p = psutil.Process()
    return p.memory_info().rss / 1024 / 1024


def clear_memory():
    gc.collect()


def run_shap_analysis(model, X_train, X_test, feature_names, out_dir='explainability_plots',
                      n_background=100, n_test_samples=500, nsamples=200, batch_size=50, max_display=30):
    os.makedirs(out_dir, exist_ok=True)
    if not SHAP_AVAILABLE:
        print('SHAP not installed or failed to import; skipping SHAP analysis.')
        return None

    print(f"Memory before SHAP: {get_memory_usage_mb():.1f} MB")

    background = shap.kmeans(X_train, n_background)
    X_test_sample = shap.sample(X_test, min(n_test_samples, len(X_test)), random_state=42)

    def predict_fn(x):
        return model.predict_proba(x)[:, 1]

    explainer = shap.KernelExplainer(predict_fn, background)

    all_shap = []
    num_batches = (len(X_test_sample) + batch_size - 1) // batch_size
    for b in range(num_batches):
        s = b * batch_size
        e = min((b + 1) * batch_size, len(X_test_sample))
        print(f'Processing SHAP batch {b+1}/{num_batches} ({s}:{e})')
        batch = X_test_sample[s:e]
        vals = explainer.shap_values(batch, nsamples=nsamples)
        all_shap.append(vals)
        clear_memory()

    shap_values = np.vstack(all_shap)
    np.save(os.path.join(out_dir, 'shap_values.npy'), shap_values)
    np.save(os.path.join(out_dir, 'X_test_sample.npy'), X_test_sample)

    # summary plot
    plt.figure(figsize=(12, max_display * 0.25))
    shap.summary_plot(shap_values, X_test_sample, feature_names=feature_names, show=False, max_display=max_display)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'shap_summary_plot.pdf'), bbox_inches='tight', dpi=300)
    plt.close()

    print('SHAP analysis done')
    return shap_values


def run_lime_analysis(model, X_train_scaled, X_test_scaled, feature_names, out_dir='explainability_plots', num_instances=3, num_features=15):
    os.makedirs(out_dir, exist_ok=True)
    if not LIME_AVAILABLE:
        print('LIME not installed or failed to import; skipping LIME analysis.')
        return None

    explainer = LimeTabularExplainer(X_train_scaled, feature_names=feature_names, class_names=['0', '1'], mode='classification', discretize_continuous=True)

    def predict_fn(X):
        return model.predict_proba(X)

    for i in range(min(num_instances, len(X_test_scaled))):
        instance = X_test_scaled[i]
        exp = explainer.explain_instance(instance, predict_fn, num_features=num_features, top_labels=2)
        
        fig = exp.as_pyplot_figure(label=1)
        plt.title(f"LIME Explanation - Instance {i+1}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'lime_explanation_instance_{i+1}.pdf'), bbox_inches='tight', dpi=300)
        plt.close()

    print('LIME analysis done')
    return True