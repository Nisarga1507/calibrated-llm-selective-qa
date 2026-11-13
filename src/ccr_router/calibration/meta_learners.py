"""Stage-1 and Stage-2 3-Signal Meta-Learners for Conformal Cascading Router."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@dataclass
class MetaLearnerBundle:
    stage1_scaler: StandardScaler
    stage1_clf: LogisticRegression
    stage2_scaler: StandardScaler
    stage2_clf: LogisticRegression


class ThreeSignalMetaLearner:
    """Fits Stage-1 (S_logit) and Stage-2 (S_logit, S_sem, S_ret) calibrated scorers."""

    def __init__(self, c_reg: float = 1.0, seed: int = 42) -> None:
        self.c_reg = c_reg
        self.seed = seed
        self.bundle: MetaLearnerBundle | None = None

    @staticmethod
    def _stage1_features(s_logit: np.ndarray) -> np.ndarray:
        s1 = np.asarray(s_logit, dtype=np.float64).reshape(-1, 1)
        p = np.clip(s1, 1e-4, 1.0 - 1e-4)
        return np.hstack([s1, np.log(p / (1.0 - p))])

    @staticmethod
    def _stage2_features(
        s_logit: np.ndarray,
        s_sem: np.ndarray,
        s_ret: np.ndarray,
    ) -> np.ndarray:
        s1 = np.asarray(s_logit, dtype=np.float64).reshape(-1, 1)
        s2 = np.asarray(s_sem, dtype=np.float64).reshape(-1, 1)
        s3 = np.asarray(s_ret, dtype=np.float64).reshape(-1, 1)
        # Include pairwise cross-signal agreement terms for hallucination detection
        return np.hstack([s1, s2, s3, s1 * s2, s1 * s3, s2 * s3])

    def fit(
        self,
        s_logit: np.ndarray,
        s_sem: np.ndarray,
        s_ret: np.ndarray,
        is_correct: np.ndarray,
    ) -> MetaLearnerBundle:
        y = np.asarray(is_correct, dtype=int)
        x1 = self._stage1_features(s_logit)
        x2 = self._stage2_features(s_logit, s_sem, s_ret)

        sc1 = StandardScaler().fit(x1)
        clf1 = LogisticRegression(C=self.c_reg, random_state=self.seed, max_iter=1000)
        clf1.fit(sc1.transform(x1), y)

        sc2 = StandardScaler().fit(x2)
        clf2 = LogisticRegression(C=self.c_reg, random_state=self.seed, max_iter=1000)
        clf2.fit(sc2.transform(x2), y)

        self.bundle = MetaLearnerBundle(
            stage1_scaler=sc1,
            stage1_clf=clf1,
            stage2_scaler=sc2,
            stage2_clf=clf2,
        )
        return self.bundle

    def predict_stage1(self, s_logit: np.ndarray) -> np.ndarray:
        if self.bundle is None:
            raise RuntimeError("MetaLearner must be fitted before prediction.")
        x1 = self._stage1_features(s_logit)
        probs = self.bundle.stage1_clf.predict_proba(self.bundle.stage1_scaler.transform(x1))[:, 1]
        return np.clip(probs, 0.001, 0.999)

    def predict_stage2(
        self,
        s_logit: np.ndarray,
        s_sem: np.ndarray,
        s_ret: np.ndarray,
    ) -> np.ndarray:
        if self.bundle is None:
            raise RuntimeError("MetaLearner must be fitted before prediction.")
        x2 = self._stage2_features(s_logit, s_sem, s_ret)
        probs = self.bundle.stage2_clf.predict_proba(self.bundle.stage2_scaler.transform(x2))[:, 1]
        return np.clip(probs, 0.001, 0.999)