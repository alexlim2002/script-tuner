"""구어성 분류기 — P(spoken) 단일 축 (cf. ADR-0014).

문어체/구어체 이진 분류기를 학습해 텍스트의 `P(spoken)`을 산출한다. 최종 평가의
headline 지표는 *같은 아이템에 대한* 변환 전/후 델타:

    delta = P(spoken | prediction) - P(spoken | input)

style_separation(casual↔semi_formal 상대 분기)과 상보적이다 — 이쪽은 "문어체 원문
대비 출력이 절대적으로 얼마나 말다워졌는가"를 본다.

피처(텍스트당, ADR-0014 §3 — 추출기는 우리 spaCy 파이프라인으로 통일):
  - filler_rate          (um/uh/you know… — ⑤ stats `_count_fillers` 재사용)
  - lexical_density      (POS, `_pos_stats`)
  - phrasal_verb_ratio   (POS, 동상)
  - pronoun_ratio        (POS, 동상)
  - tokens_per_sentence  (우리 분절 = pause strip 후 종결부호; style_separation과 동일)
  - contraction_rate     (don't/I'm/gonna… 축약형)

★ pause_rate는 **의도적으로 제외**한다. `<pause:*>`는 전사 주석 기호라(spoken_text엔
있고 formal_text엔 없음, cf. ADR-0008) 그걸 세면 "구어성"이 아니라 주석 출처를 학습해
분리도가 가짜로 부풀려진다(.work/note-spokenness-classifier-findings.md). filler는
실제 발화 단어이므로 포함한다 — 그 구분이 본 피처셋의 핵심.

순환성 통제(ADR-0014 §4): 분류기는 모델 train split 페어로만 학습하고, 델타는
모델·분류기 모두에게 held-out인 test split predictions에서만 측정한다. 호출자가
`splits/train.jsonl` 페어를 넘기는 것으로 이 규약을 강제한다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from scripttuner.preprocessing.ir import Pair
from scripttuner.preprocessing.stats import _load_spacy, _pos_stats
from scripttuner.training.style_separation import _surface_features

if TYPE_CHECKING:
    from spacy.language import Language

# 피처 벡터 순서 — 학습·추론에서 동일하게 유지.
FEATURE_NAMES: tuple[str, ...] = (
    "filler_rate",
    "lexical_density",
    "phrasal_verb_ratio",
    "pronoun_ratio",
    "tokens_per_sentence",
    "contraction_rate",
)

# 축약형 — 아포스트로피 접미(n't/'s/'re/'ve/'ll/'m/'d) + 비격식 융합형.
_CONTRACTION_RE = re.compile(
    r"\b\w+'(?:t|s|re|ve|ll|m|d)\b|\b(?:gonna|wanna|gotta|kinda|gimme|lemme|dunno)\b",
    re.IGNORECASE,
)

# spoken=1 / formal=0 라벨 규약(분류기 양성 클래스 = spoken).
_LABEL_SPOKEN = 1
_LABEL_FORMAL = 0


@dataclass(frozen=True)
class SpokennessModel:
    """학습된 구어성 분류기 — scaler + LogisticRegression + 피처 순서를 함께 보관한다."""

    scaler: Any
    classifier: Any
    feature_names: tuple[str, ...]


def extract_features(text: str, nlp: Language) -> dict[str, float]:
    """텍스트에서 구어성 피처를 추출한다(FEATURE_NAMES)."""
    sf = _surface_features(text)  # filler_rate, tokens_per_sentence, tokens(분모)
    pos = _pos_stats(text, nlp)
    tokens = sf["tokens"] or 1.0
    return {
        "filler_rate": sf["filler_rate"],
        "lexical_density": pos["lexical_density"],
        "phrasal_verb_ratio": pos["phrasal_verb_ratio"],
        "pronoun_ratio": pos["pronoun_ratio"],
        "tokens_per_sentence": sf["tokens_per_sentence"],
        "contraction_rate": len(_CONTRACTION_RE.findall(text)) / tokens,
    }


def _feature_vector(text: str, nlp: Language) -> list[float]:
    f = extract_features(text, nlp)
    return [f[name] for name in FEATURE_NAMES]


def _feature_matrix(texts: list[str], nlp: Language) -> list[list[float]]:
    return [_feature_vector(t, nlp) for t in texts]


def train(
    train_pairs: list[Pair],
    *,
    seed: int = 42,
    nlp: Language | None = None,
) -> tuple[SpokennessModel, dict[str, Any]]:
    """train split 페어로 구어성 분류기를 학습한다.

    각 페어에서 spoken_text→label 1, formal_text→label 0. 분류기 *자체* 품질을
    재기 위해 내부 stratified holdout(80/20)에서 accuracy/AUC를 산출한 뒤, 최종
    영속 모델은 전체 train 페어로 다시 적합한다.

    모델은 LogisticRegression(StandardScaler 선행) — 표면 피처 관계가 대체로
    선형이고, 부호 있는 계수가 "어떤 피처가 spoken 쪽인가"를 해석 가능하게 준다
    (cf. .work/note-spokenness-classifier-findings.md §3).

    반환: (학습된 모델, 품질 메트릭 dict).
    """
    if not train_pairs:
        raise ValueError("train_pairs must be non-empty")

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    nlp = nlp or _load_spacy()

    texts: list[str] = []
    labels: list[int] = []
    for p in train_pairs:
        texts.append(p.spoken_text or "")
        labels.append(_LABEL_SPOKEN)
        texts.append(p.formal_text or "")
        labels.append(_LABEL_FORMAL)

    X = _feature_matrix(texts, nlp)

    def _fit(features: list[list[float]], y: list[int]) -> tuple[Any, Any]:
        scaler = StandardScaler().fit(features)
        clf = LogisticRegression(max_iter=1000, random_state=seed)
        clf.fit(scaler.transform(features), y)
        return scaler, clf

    # --- 내부 holdout으로 분류기 품질 측정 ---
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, labels, test_size=0.2, random_state=seed, stratify=labels
    )
    scaler_eval, clf_eval = _fit(X_tr, y_tr)
    X_te_s = scaler_eval.transform(X_te)
    spoken_col = list(clf_eval.classes_).index(_LABEL_SPOKEN)
    proba_te = clf_eval.predict_proba(X_te_s)[:, spoken_col]
    preds_te = clf_eval.predict(X_te_s)
    metrics: dict[str, Any] = {
        "stage": "spokenness_train",
        "n_pairs": len(train_pairs),
        "n_samples": len(texts),
        "holdout_accuracy": float(accuracy_score(y_te, preds_te)),
        "holdout_auc": float(roc_auc_score(y_te, proba_te)),
        # 부호 있는 계수: 양수 = spoken 쪽으로 미는 피처(StandardScaler 후 기준).
        "coefficients": dict(
            zip(FEATURE_NAMES, (float(c) for c in clf_eval.coef_[0]), strict=True)
        ),
    }

    # --- 최종 모델: 전체 train 페어로 재적합 ---
    scaler, clf = _fit(X, labels)
    return SpokennessModel(scaler=scaler, classifier=clf, feature_names=FEATURE_NAMES), metrics


def score(texts: list[str], model: SpokennessModel, *, nlp: Language | None = None) -> list[float]:
    """텍스트 리스트의 P(spoken)을 반환한다."""
    if not texts:
        return []
    nlp = nlp or _load_spacy()
    X = _feature_matrix(texts, nlp)
    X_s = model.scaler.transform(X)
    spoken_col = list(model.classifier.classes_).index(_LABEL_SPOKEN)
    return [float(p) for p in model.classifier.predict_proba(X_s)[:, spoken_col]]


def save(model: SpokennessModel, path: Path) -> None:
    """학습된 모델을 joblib으로 영속화한다."""
    import joblib

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load(path: Path) -> SpokennessModel:
    """영속화된 모델을 로드한다."""
    import joblib

    model: SpokennessModel = joblib.load(path)
    return model
