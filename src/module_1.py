
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import torch
from sklearn.pipeline import Pipeline

try:
    shap.initjs()  # only matters in notebooks; harmless to skip in Streamlit/scripts
except Exception:
    pass

# =============================================================================
# SECTION 0 — CONFIG (edit these to match your variable names)
# =============================================================================

OUTPUT_DIR = "shap_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Background sample for KernelExplainer (autoencoder) -- keep small (50-100 rows)
# because KernelExplainer is expensive.
AE_BACKGROUND_SIZE = 50
AE_EXPLAIN_SAMPLE_SIZE = 100  # how many rows to compute AE SHAP values for (bar plot)


# =============================================================================
# SECTION 0.5 — PIPELINE HELPERS
# TreeExplainer cannot take an sklearn Pipeline ("Model type not yet supported
# by TreeExplainer: sklearn.pipeline.Pipeline"). These helpers peel the
# Pipeline apart: the preprocessing steps are applied to the data manually and
# only the final XGBoost estimator is handed to SHAP.
# =============================================================================

def _unwrap_model(model):
    """Return (list_of_preprocessing_pipelines, final_tree_estimator)."""
    pre_steps = []
    current = model
    while isinstance(current, Pipeline):
        if len(current.steps) > 1:
            pre_steps.append(current[:-1])
        current = current.steps[-1][1]
    return pre_steps, current


def make_xgb_explainer(xgb_model):
    """
    Build a TreeExplainer from an XGBClassifier OR a sklearn Pipeline ending in one.
    Use this instead of shap.TreeExplainer(model) so Pipelines don't crash.
    """
    _, final_estimator = _unwrap_model(xgb_model)
    return shap.TreeExplainer(final_estimator)


def _prepare_features(model, X):
    """
    Push X through every preprocessing step of the model.

    Returns
    -------
    final_estimator : the bare tree model (what TreeExplainer needs)
    X_model         : DataFrame exactly as the final estimator sees it
    X_display       : DataFrame used for plot labels/colours. If the pipeline
                      only rescales columns (same names, same order) this is the
                      ORIGINAL unscaled data, so beeswarm colours and waterfall
                      values read in real units (e.g. Dst Port = 22).
    """
    X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    pre_steps, final_estimator = _unwrap_model(model)

    Xt = X
    for pre in pre_steps:
        out = pre.transform(Xt)
        if hasattr(out, "toarray"):          # sparse -> dense
            out = out.toarray()
        if isinstance(out, pd.DataFrame):
            arr, names = out.values, list(out.columns)
        else:
            arr = np.asarray(out)
            try:
                names = [str(n).split("__")[-1] for n in pre.get_feature_names_out()]
            except Exception:
                names = (list(Xt.columns) if arr.shape[1] == Xt.shape[1]
                         else [f"feature_{i}" for i in range(arr.shape[1])])
        Xt = pd.DataFrame(arr, columns=names, index=X.index)

    Xt = Xt.astype(float)
    X_display = X if list(Xt.columns) == list(X.columns) else Xt
    return final_estimator, Xt, X_display


def _explain(explainer, X_model, X_display):
    """Run the explainer on model-space data, attach display-space data/names."""
    exp = explainer(X_model.values, check_additivity=False)
    return shap.Explanation(
        values=exp.values,
        base_values=exp.base_values,
        data=X_display.values,
        feature_names=list(X_display.columns),
    )



# =============================================================================
# SECTION 0.6 — SAMPLE SELECTION HELPERS
# Positions are 0-based row positions (iloc) in X_test, so the valid range is
# 0 .. len(X_test)-1.
# =============================================================================

def _check_index(idx, n, what="sample_index"):
    if idx is None or not (0 <= int(idx) < n):
        raise IndexError(f"{what}={idx} is out of range. Valid range: 0 to {n - 1} "
                         f"(the dataset has {n} samples).")
    return int(idx)


def _label_name(v, class_names=None):
    """Turn an int-encoded or string label into a readable name."""
    try:
        if class_names is not None and float(v) == int(v):
            return str(class_names[int(v)])
    except (TypeError, ValueError, IndexError):
        pass
    return str(v)


# =============================================================================
# SECTION 1 — TIER 1: XGBoost (Supervised) — Global + Local Explanations
# =============================================================================

def explain_xgboost_global(xgb_model, X_test, class_names=None, save=True, title_suffix=""):
    """
    SHAP Summary Plot (Beeswarm) -- Global Model Evaluation Section.

    Proves the model isn't 'cheating': shows that high values (red) of
    features like 'Dst Port' (e.g. ports 21/22) push predictions toward
    specific classes like 'Patator'.
    """
    final_est, X_model, X_disp = _prepare_features(xgb_model, X_test)
    explainer = make_xgb_explainer(xgb_model)
    shap_values = _explain(explainer, X_model, X_disp)

    # For multi-class XGBoost, shap_values.values has shape
    # (n_samples, n_features, n_classes). We plot one beeswarm per class.
    if shap_values.values.ndim == 3:
        n_classes = shap_values.values.shape[2]
        labels = class_names if class_names is not None else [f"class_{i}" for i in range(n_classes)]

        for i, cls in enumerate(labels):
            plt.figure()
            shap.summary_plot(
                shap_values.values[:, :, i],
                X_disp,
                show=False,
                plot_type="dot",
            )
            plt.title(f"SHAP Summary (Beeswarm) — XGBoost — Class: {cls}{title_suffix}")
            if save:
                plt.savefig(f"{OUTPUT_DIR}/xgb_summary_beeswarm_{cls}.png", bbox_inches="tight", dpi=150)
            plt.close()
    else:
        # Binary case
        plt.figure()
        shap.summary_plot(shap_values.values, X_disp, show=False, plot_type="dot")
        plt.title(f"SHAP Summary (Beeswarm) — XGBoost{title_suffix}")
        if save:
            plt.savefig(f"{OUTPUT_DIR}/xgb_summary_beeswarm.png", bbox_inches="tight", dpi=150)
        plt.close()

    print(f"[Tier 1] Beeswarm plot(s) saved to {OUTPUT_DIR}/")
    return explainer, shap_values


def explain_xgboost_case_study(xgb_model, explainer, X_test, y_test,
                                target_class="DoS", sample_index=None, save=True,
                                class_names=None, display_index=None):
    """
    SHAP Waterfall Plot -- Case Study Section for a confirmed known attack.

    Picks a sample of `target_class` (preferring one the model also predicts
    as that class) and shows the step-by-step breakdown of how features
    pushed the prediction from the baseline to the final label.

    `class_names` (optional): list mapping class index -> name. Lets the
    function match `target_class` when y_test is integer-encoded.
    """
    final_est, X_model, X_disp = _prepare_features(xgb_model, X_test)
    preds = np.asarray(final_est.predict(X_model.values)).astype(int)
    names = list(class_names) if class_names is not None else None

    if sample_index is not None:
        sample_index = _check_index(sample_index, len(X_model))   # manual pick: 0..N-1

    if sample_index is None:
        y = np.asarray(y_test)
        true_mask = (y == target_class)
        target_idx = names.index(target_class) if names and target_class in names else None
        if not true_mask.any() and target_idx is not None:
            true_mask = (y == target_idx)          # y_test is integer-encoded

        if target_idx is not None:
            both = np.where(true_mask & (preds == target_idx))[0]
            candidates = both if len(both) else np.where(true_mask)[0]
        else:
            candidates = np.where(true_mask)[0]

        if len(candidates) == 0:
            raise ValueError(f"No samples found for class '{target_class}'. Pass sample_index explicitly.")
        sample_index = int(candidates[0])

    # display_index: the row number to SHOW in title/filename (e.g. position in the
    # full dataset when X_test only holds that one row). Defaults to sample_index.
    shown = sample_index if display_index is None else int(display_index)

    row_expl = _explain(explainer, X_model.iloc[[sample_index]], X_disp.iloc[[sample_index]])

    if row_expl.values.ndim == 3:
        pred_class_idx = int(preds[sample_index])
        single_explanation = shap.Explanation(
            values=row_expl.values[0, :, pred_class_idx],
            base_values=row_expl.base_values[0, pred_class_idx],
            data=row_expl.data[0],
            feature_names=row_expl.feature_names,
        )
    else:
        single_explanation = row_expl[0]

    plt.figure()
    shap.plots.waterfall(single_explanation, show=False)
    _pred_name = _label_name(preds[sample_index], names)
    plt.title(f"SHAP Waterfall — XGBoost — Sample #{shown} (Predicted: {_pred_name})")
    if save:
        plt.savefig(f"{OUTPUT_DIR}/xgb_waterfall_sample_{shown}.png", bbox_inches="tight", dpi=150)
    plt.close()

    print(f"[Tier 1] Waterfall plot for sample {shown} saved to {OUTPUT_DIR}/")
    return single_explanation


# =============================================================================
# SECTION 2 — TIER 2: Autoencoder (Unsupervised / Zero-Day)
# =============================================================================

def reconstruction_error_fn(autoencoder, device=None, batch_size=256):
    """
    Wraps the PyTorch autoencoder so SHAP can treat 'reconstruction error'
    as the model's scalar output. This is the key trick: autoencoders don't
    predict a class, so we explain the ERROR itself.

    Works with any nn.Module whose forward(x) returns the reconstruction
    (same shape as input). Runs in eval mode with no_grad, batched so
    KernelExplainer's repeated calls (thousands of rows) don't blow up
    memory or run through the autograd graph.
    """
    device = device or next(autoencoder.parameters()).device
    autoencoder.eval()

    def predict_fn(X):
        X = np.asarray(X, dtype=np.float32)
        errors = np.empty(len(X), dtype=np.float32)

        with torch.no_grad():
            for start in range(0, len(X), batch_size):
                end = start + batch_size
                batch = torch.from_numpy(X[start:end]).to(device)
                reconstructed = autoencoder(batch)
                # squared error per sample, averaged over features
                mse = torch.mean((batch - reconstructed) ** 2, dim=1)
                errors[start:end] = mse.cpu().numpy()

        return errors

    return predict_fn


def make_ae_explainer(autoencoder, X_background, background_size=AE_BACKGROUND_SIZE):
    """
    Build the KernelExplainer for the autoencoder's reconstruction error.
    Cheap to create (it only stores a small background sample), so any plot that
    needs it can build it on demand -- no other plot has to be generated first.
    """
    predict_fn = reconstruction_error_fn(autoencoder)
    background = shap.sample(X_background, background_size, random_state=42)
    return shap.KernelExplainer(predict_fn, background)


FORCE_PLOT_DECIMALS = 3      # decimals shown for feature values in force-plot labels (3-4 works well)


def _sig(v, decimals=FORCE_PLOT_DECIMALS):
    """Round a feature value to a few decimals so force-plot labels stay short."""
    try:
        return round(float(v), decimals)
    except (TypeError, ValueError):
        return v


def _draw_force_plot(base_value, values, features, feature_names, title, path=None):
    """
    Matplotlib force plot without overlapping text:
      * feature values rounded to FORCE_PLOT_DECIMALS decimals (1153.9466666666665 -> 1153.947)
      * labels rotated so neighbouring labels cannot collide
      * title raised above the 'base value' / f(x) captions
    Only this figure is affected (no global matplotlib settings are changed).
    """
    feats = np.array([_sig(v) for v in np.asarray(features, dtype=float)], dtype=float)
    try:
        shap.force_plot(base_value, np.asarray(values), feats, feature_names=list(feature_names),
                        matplotlib=True, show=False, figsize=(20, 3), text_rotation=25)
    except TypeError:       # older SHAP without figsize / text_rotation
        shap.force_plot(base_value, np.asarray(values), feats, feature_names=list(feature_names),
                        matplotlib=True, show=False)
    fig = plt.gcf()
    _declutter_force_plot(fig, title)
    if path:
        fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def _declutter_force_plot(fig, title):
    """
    1) If the 'base value' and 'f(x)' captions land on top of each other (base value ~ output),
       push them apart.
    2) Place the title just above the highest element of the plot (the 'higher / lower'
       legend), so a long title can never run into it.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax = fig.axes[0] if fig.axes else None
    if ax is not None:
        base = next((t for t in ax.texts if t.get_text().strip().lower().startswith("base value")), None)
        fx = next((t for t in ax.texts if t.get_text().strip() == "f(x)"), None)
        fx_val = None
        if fx is not None:
            fx_val = next((t for t in ax.texts if t is not fx and t.get_fontweight() in ("bold", 700)
                           and abs(t.get_position()[0] - fx.get_position()[0]) < 1e-9
                           and t.get_text().replace(".", "").replace("-", "").strip().isdigit()), None)
        if base is not None and fx is not None:
            b1, b2 = base.get_window_extent(renderer), fx.get_window_extent(renderer)
            if b1.overlaps(b2) or abs(b1.x1 - b2.x0) < 8 or abs(b2.x1 - b1.x0) < 8:
                inv = ax.transData.inverted()
                gap_px = (min(b1.x1, b2.x1) - max(b1.x0, b2.x0)) / 2 + 12   # half the overlap plus padding
                gap_px = max(gap_px, 20)
                for t, sign, ha in ((base, -1, "right"), (fx, 1, "left"), (fx_val, 1, "left")):
                    if t is None:
                        continue
                    x0, y0 = t.get_position()
                    px = ax.transData.transform((x0, y0))
                    nx = inv.transform((px[0] + sign * gap_px, px[1]))[0]
                    t.set_position((nx, y0)); t.set_ha(ha)
        fig.canvas.draw()
    tb = fig.get_tightbbox(renderer)
    fig_h = fig.get_size_inches()[1]
    fig.suptitle(title, y=(tb.y1 + 0.18) / fig_h, va="bottom", fontsize=12, fontweight="bold")


def explain_autoencoder_global(autoencoder, X_train, X_test,
                                background_size=AE_BACKGROUND_SIZE,
                                sample_size=AE_EXPLAIN_SAMPLE_SIZE,
                                save=True, title_suffix=""):
    """
    SHAP Bar Plot -- Global Feature Importance for Anomaly.

    Placed next to your threshold visualization graph. Averages all
    anomalies to show WHICH network features are generally responsible
    for breaking the autoencoder's expectations (i.e. driving up
    reconstruction error).
    """
    # Small background set (KernelExplainer scales poorly with size)
    explainer = make_ae_explainer(autoencoder, X_train, background_size)

    sample = X_test.sample(min(sample_size, len(X_test)), random_state=42)
    shap_values = explainer.shap_values(sample, nsamples="auto")

    plt.figure()
    shap.summary_plot(shap_values, sample, plot_type="bar", show=False)
    plt.title(f"SHAP Global Feature Importance — Autoencoder (Reconstruction Error){title_suffix}")
    if save:
        plt.savefig(f"{OUTPUT_DIR}/ae_global_bar.png", bbox_inches="tight", dpi=150)
    plt.close()

    print(f"[Tier 2] Autoencoder global bar plot saved to {OUTPUT_DIR}/")
    return explainer, shap_values, sample


def explain_zero_day_proof(xgb_model, xgb_explainer, autoencoder, ae_explainer,
                            X_train, X_test, reconstruction_threshold,
                            zero_day_index=None, class_names=None, save=True,
                            X_test_ae=None, display_index=None):
    """
    SHAP Force Plots -- THE Zero-Day Proof.

    Finds a sample where XGBoost says 'Benign' but the Autoencoder flags
    it as an anomaly (reconstruction error > threshold). Produces TWO
    force plots:
        1. XGBoost force plot  -> pushes toward "Benign"
        2. Autoencoder force plot -> big red arrow pushing error past threshold

    `xgb_explainer` and `ae_explainer` may be None -- they are then built here,
    so this plot does not depend on the beeswarm / bar plot having been run.

    This visually tells the story:
        "XGBoost thought this was normal, but the Autoencoder flagged it
         because this specific combination of features was structurally alien."
    """
    # X_test       -> features as XGBoost expects them (raw; pipeline is applied inside)
    # X_test_ae    -> features as the autoencoder expects them (scaled, its own columns).
    #                 Same rows, same order. Defaults to X_test if both models share features.
    ae_frame = X_test if X_test_ae is None else X_test_ae

    # Independent of every other plot: build the explainers here if none were passed in.
    if xgb_explainer is None:
        xgb_explainer = make_xgb_explainer(xgb_model)
    if ae_explainer is None:
        ae_explainer = make_ae_explainer(autoencoder, X_train)

    predict_fn = reconstruction_error_fn(autoencoder)
    errors = predict_fn(ae_frame.values)

    final_est, X_model, X_disp = _prepare_features(xgb_model, X_test)
    xgb_preds = np.asarray(final_est.predict(X_model.values)).astype(int)

    if class_names is None:
        benign_label = 0
    else:
        names_lower = [str(c).lower() for c in class_names]   # 'BENIGN' / 'Benign' both work
        benign_label = names_lower.index("benign")

    if zero_day_index is None:
        candidates = np.where((xgb_preds == benign_label) & (errors > reconstruction_threshold))[0]
        if len(candidates) == 0:
            raise ValueError("No disagreement samples found (XGBoost=Benign, AE=Anomaly). "
                              "Pass zero_day_index manually or check your threshold.")
        zero_day_index = int(candidates[0])
    else:
        # Manual pick: must be a valid row AND flagged by the autoencoder.
        zero_day_index = _check_index(zero_day_index, len(errors), "zero_day_index")
        if errors[zero_day_index] <= reconstruction_threshold:
            raise ValueError(
                f"Sample {zero_day_index} is NOT flagged as an anomaly "
                f"(error {errors[zero_day_index]:.5f} <= threshold {reconstruction_threshold:.5f}). "
                f"Pick one from find_anomaly_samples().")

    shown = zero_day_index if display_index is None else int(display_index)
    xgb_pred_idx = int(xgb_preds[zero_day_index])
    xgb_pred_name = _label_name(xgb_pred_idx, class_names)

    sample_row_ae = ae_frame.iloc[[zero_day_index]]

    # --- XGBoost force plot (pushing toward Benign) ---
    row_expl = _explain(xgb_explainer,
                        X_model.iloc[[zero_day_index]],
                        X_disp.iloc[[zero_day_index]])
    if row_expl.values.ndim == 3:
        base_value = row_expl.base_values[0, xgb_pred_idx]   # explain the class XGBoost actually chose
        values = row_expl.values[0, :, xgb_pred_idx]
    else:
        base_value = row_expl.base_values[0]
        values = row_expl.values[0]

    _draw_force_plot(
        base_value, values, row_expl.data[0], row_expl.feature_names,
        f"XGBoost Force Plot — Sample #{shown} (Predicted: {xgb_pred_name})",
        f"{OUTPUT_DIR}/zero_day_xgb_force_{shown}.png" if save else None)

    # --- Autoencoder force plot (pushing error past threshold) ---
    ae_shap_vals = ae_explainer.shap_values(sample_row_ae, nsamples="auto")
    ae_base_value = ae_explainer.expected_value

    _draw_force_plot(
        ae_base_value, ae_shap_vals[0], sample_row_ae.iloc[0].values, list(sample_row_ae.columns),
        f"Autoencoder Force Plot — Sample #{shown} (Reconstruction Error > Threshold = Zero-Day Anomaly)",
        f"{OUTPUT_DIR}/zero_day_ae_force_{shown}.png" if save else None)

    print(f"[Zero-Day Proof] Force plots for sample {shown} saved to {OUTPUT_DIR}/")
    print(f"  -> XGBoost predicted: {xgb_pred_name}")
    print(f"  -> Autoencoder reconstruction error: {errors[zero_day_index]:.5f} "
          f"(threshold: {reconstruction_threshold:.5f})")

    return zero_day_index
