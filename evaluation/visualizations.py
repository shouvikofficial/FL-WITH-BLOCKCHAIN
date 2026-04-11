import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — avoids GUI/threading issues
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

# ── Dark theme colors (matching the Watchtower dashboard) ──────────────────
BG_COLOR   = '#0f172a'   # Deep navy background
CARD_BG    = '#1e293b'   # Card background
TEXT_COLOR  = '#e5e7eb'   # Light text
MUTED_TEXT  = '#9ca3af'   # Muted / axis text
GRID_COLOR  = '#374151'   # Grid lines
ACCENT_PINK = '#ec4899'   # ROC curve
ACCENT_INDIGO = '#6366f1' # Reference line
ACCENT_GREEN  = '#10b981' # PR curve


def _apply_dark_style(ax, title_text):
    """Apply consistent dark styling to any axes."""
    ax.set_facecolor(BG_COLOR)
    ax.set_title(title_text, pad=15, color=TEXT_COLOR, fontsize=14)
    ax.tick_params(colors=MUTED_TEXT, labelsize=10)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)


def generate_visualizations(model, X, y, output_dir="static", feature_names=None):
    """
    Generate Advanced Analytics plots for the Server Dashboard.
    Plots are saved directly to the stated output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Generate Predictions
    y_pred = model.predict(X)

    # Probability (for ROC and SHAP if needed)
    y_proba = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X)[:, 1]
        except Exception:
            pass

    # ──────────────────────────────────────────────────────────────
    # 1. Confusion Matrix
    # ──────────────────────────────────────────────────────────────
    try:
        cm = confusion_matrix(y, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5), facecolor=BG_COLOR)
        sns.heatmap(cm, annot=True, fmt='d', cmap='mako', cbar=False,
                     annot_kws={"size": 16, "color": TEXT_COLOR},
                     linewidths=1, linecolor=GRID_COLOR, ax=ax)
        ax.set_facecolor(CARD_BG)
        ax.set_title('Confusion Matrix', pad=15, color=TEXT_COLOR, fontsize=14)
        ax.set_xlabel('Predicted Label', color=MUTED_TEXT, fontsize=12)
        ax.set_ylabel('True Label', color=MUTED_TEXT, fontsize=12)
        ax.tick_params(colors=MUTED_TEXT, labelsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'confusion_matrix.png'),
                    dpi=150, facecolor=BG_COLOR, bbox_inches='tight')
        plt.close(fig)
        print("    ✅ Confusion Matrix saved.")
    except Exception as e:
        print(f"    ❌ Confusion Matrix failed: {e}")
        plt.close('all')

    # ──────────────────────────────────────────────────────────────
    # 2. ROC-AUC Curve
    # ──────────────────────────────────────────────────────────────
    if y_proba is not None:
        try:
            fpr, tpr, _ = roc_curve(y, y_proba)
            roc_auc = auc(fpr, tpr)

            fig, ax = plt.subplots(figsize=(6, 5), facecolor=BG_COLOR)
            ax.plot(fpr, tpr, color=ACCENT_PINK, lw=2,
                    label=f'ROC Curve (AUC = {roc_auc:.3f})')
            ax.plot([0, 1], [0, 1], color=ACCENT_INDIGO, lw=2, linestyle='--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate', color=MUTED_TEXT)
            ax.set_ylabel('True Positive Rate', color=MUTED_TEXT)
            ax.legend(loc="lower right", facecolor=CARD_BG,
                      edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
            ax.grid(color=GRID_COLOR, linestyle='--', linewidth=0.5, alpha=0.5)
            ax.set_facecolor(BG_COLOR)
            _apply_dark_style(ax, 'Receiver Operating Characteristic (ROC)')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, 'roc_curve.png'),
                        dpi=150, facecolor=BG_COLOR, bbox_inches='tight')
            plt.close(fig)
            print("    ✅ ROC Curve saved.")
        except Exception as e:
            print(f"    ❌ ROC Curve failed: {e}")
            plt.close('all')

        # ──────────────────────────────────────────────────────────
        # 3. Precision-Recall Curve
        # ──────────────────────────────────────────────────────────
        try:
            precision, recall, _ = precision_recall_curve(y, y_proba)

            fig, ax = plt.subplots(figsize=(6, 5), facecolor=BG_COLOR)
            ax.plot(recall, precision, color=ACCENT_GREEN, lw=2,
                    label='Precision-Recall')
            ax.set_xlabel('Recall', color=MUTED_TEXT)
            ax.set_ylabel('Precision', color=MUTED_TEXT)
            ax.legend(loc="lower left", facecolor=CARD_BG,
                      edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
            ax.grid(color=GRID_COLOR, linestyle='--', linewidth=0.5, alpha=0.5)
            ax.set_facecolor(BG_COLOR)
            _apply_dark_style(ax, 'Precision-Recall Curve')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, 'pr_curve.png'),
                        dpi=150, facecolor=BG_COLOR, bbox_inches='tight')
            plt.close(fig)
            print("    ✅ Precision-Recall Curve saved.")
        except Exception as e:
            print(f"    ❌ Precision-Recall Curve failed: {e}")
            plt.close('all')

    # ──────────────────────────────────────────────────────────────
    # 4. SHAP Feature Importance Summary Plot
    # ──────────────────────────────────────────────────────────────
    try:
        import shap

        sample_size = min(50, len(X))
        X_sample = X[:sample_size]

        def model_predict(data):
            if hasattr(model, 'predict_proba'):
                return model.predict_proba(data)[:, 1]
            return model.predict(data)

        explainer = shap.KernelExplainer(model_predict, shap.kmeans(X_sample, 10))
        shap_values = explainer.shap_values(X_sample, silent=True)

        # Compute mean absolute SHAP values manually for a clean bar chart
        # Use real feature names if provided, otherwise fallback to generic
        if feature_names is not None and len(feature_names) == X_sample.shape[1]:
            feat_names = feature_names
        else:
            feat_names = [f"Feature {i+1}" for i in range(X_sample.shape[1])]
        mean_shap = np.abs(np.array(shap_values)).mean(axis=0)

        # Sort descending
        sorted_idx = np.argsort(mean_shap)[::-1][:15]  # top 15
        sorted_vals = mean_shap[sorted_idx]
        sorted_names = [feat_names[i] for i in sorted_idx]

        fig, ax = plt.subplots(figsize=(7, 5), facecolor=BG_COLOR)
        ax.barh(range(len(sorted_idx)), sorted_vals[::-1],
                color=ACCENT_PINK, edgecolor=GRID_COLOR)
        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels(sorted_names[::-1])
        ax.set_xlabel('Mean |SHAP Value|', color=MUTED_TEXT)
        ax.set_facecolor(BG_COLOR)
        _apply_dark_style(ax, 'SHAP Feature Significance')
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, 'shap_summary.png'),
                    dpi=150, facecolor=BG_COLOR, bbox_inches='tight')
        plt.close(fig)
        print("    ✅ SHAP Summary saved.")
    except Exception as e:
        print(f"    ⚠️ SHAP plot skipped: {e}")
        plt.close('all')
