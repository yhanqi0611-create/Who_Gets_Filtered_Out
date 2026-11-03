"""Logistic regression: pass_0.9 ~ has_dialect_ref + log(dur) + split."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def fit_pass_logit(
    has_dialect: np.ndarray,
    passed: np.ndarray,
    dur: np.ndarray,
    splits: np.ndarray,
) -> dict[str, Any]:
    df = pd.DataFrame(
        {
            "pass_09": passed.astype(int),
            "has_dialect_ref": has_dialect.astype(int),
            "log_dur": np.log(np.clip(dur.astype(float), 1e-6, None)),
            "split": splits.astype(str),
        }
    )
    formula = "pass_09 ~ has_dialect_ref + log_dur"
    if df["split"].nunique() >= 2:
        formula += " + C(split)"
    try:
        model = smf.logit(formula, data=df).fit(disp=False, maxiter=200)
        key = "has_dialect_ref"
        coef = float(model.params[key])
        ci = model.conf_int().loc[key]
        or_hat = float(np.exp(coef))
        or_lo, or_hi = float(np.exp(float(ci.iloc[0]))), float(np.exp(float(ci.iloc[1])))
        p = float(model.pvalues[key])
        if not np.isfinite(or_hat):
            or_hat, or_lo, or_hi = float("nan"), float("nan"), float("nan")
        return {
            "n": int(len(df)),
            "or": or_hat,
            "or_ci_lo": or_lo,
            "or_ci_hi": or_hi,
            "p": p,
            "coef": coef,
        }
    except Exception as exc:  # perfect separation on tiny toys
        return {
            "n": int(len(df)),
            "or": float("nan"),
            "or_ci_lo": float("nan"),
            "or_ci_hi": float("nan"),
            "p": float("nan"),
            "coef": float("nan"),
            "error": str(exc),
        }
