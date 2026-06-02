# ADR-0014: 분류기 기반 구어성 점수 메트릭 (P(spoken) 입력→출력 델타)

- **Status**: Accepted
- **Date**: 2026-06-03

## Context

현 평가는 두 축으로 구성된다:

- `style_separation.py`([ADR-0013](0013-semi-formal-data-sourcing.md) §6) — casual ↔
  semi_formal 출력이 *서로 갈라지는지*(상대적 분기)를 Cohen's d로 측정.
- `evaluate.py` — prediction의 구어성 피처 분포가 reference에 *정렬되는지*를 같은
  척도로 비교.

둘 다 **"입력 대비 출력이 절대적으로 얼마나 말다워졌는가"**는 보여주지 못한다.
style_separation은 두 스타일이 갈라진다는 것만, evaluate는 분포 정렬만 본다. 최종
보고서에는 "문어체 원문 → 변환 출력"의 구어성 상승을 단일 축으로 제시할 지표가 없다.

팀원이 SBC 페어(`spoken_text`/`formal_text`)로 문어체/구어체 이진 분류기(RF/SVM,
NLTK 4피처)를 시제작했다. 이 분류기의 P(spoken)을 단일 구어성 축으로 쓰면 위 공백을
메울 수 있다. `style_separation.py` docstring도 *"분류기 기반 separability(AUC)는
보류 — 제안서 sklearn 진단 모듈과 합류 여지"*로 이 자리를 열어두었다.

이 ADR은 그 분류기 점수를 **정식 headline 구어성 지표**로 채택하는 결정과, 채택 시
구성 타당성을 위협하는 순환성을 어떻게 통제할지를 닫는다.

## Decision

### 1. 지표 정의 — 입력→출력 P(spoken) 델타

구어성 = 학습된 분류기의 `P(spoken)`. headline 지표는 **같은 아이템에 대한 변환
전/후 델타**:

> `delta = P(spoken | prediction) − P(spoken | input)`

`predictions.jsonl`의 `input`(모델 입력=문어체스러운 원문)과 `prediction`(변환 출력)에서
산출한다. style_separation(스타일 간 상대 분기)과 **상보적**이며 대체하지 않는다.

### 2. 점수 모델 — LogisticRegression `predict_proba` (StandardScaler 선행)

보정된 확률 축이 필요하다. 표면 피처 관계가 대체로 선형이라 LogReg가 RF보다 약간
낫고(실측 auc 0.71→0.74), **부호 있는 계수**가 "어떤 피처가 spoken 쪽인가"를 바로
해석하게 해준다(예: filler↑→spoken). 팀원 노트북의 RF/SVM·시각화는 **그들의 기여로
별도 인용**하고, 우리는 점수 축만 scripttuner에 재구현한다(scaler·model 영속화).
※ 당초 RF를 적었으나 실측 후 LogReg로 교체(아래 Update).

### 3. 피처 — 우리 spaCy 파이프라인 + filler 포함, pause 제외

추출기는 우리 spaCy 기반으로 통일한다(팀원은 NLTK; NLTK `sent_tokenize`는 우리
구어 텍스트의 pause·구두점 부재에서 분절을 틀린다). 피처:
`filler_rate`, `lexical_density`, `phrasal_verb_ratio`, `pronoun_ratio`,
`tokens_per_sentence`, `contraction_rate`.

**`pause_rate`는 의도적으로 제외**한다 — `<pause:*>`는 전사 주석 기호라(spoken엔
있고 formal엔 없음) 그걸 세면 register가 아니라 *주석 출처*를 학습한다. 반면
`filler`(um/uh/you know)는 실제 발화 단어라 정당한 신호로 포함한다. 이 구분이
핵심이다(근거·실측: 아래 Update, `.work/note-spokenness-classifier-findings.md`).

### 4. 순환성 통제 — 분류기 train ⊆ 모델 train, 델타는 공통 test split

분류기의 label 1(spoken)은 모델의 학습 타깃과 같은 분포다. 같은 페어로 분류기를
학습하고 그 페어의 출력을 평가하면, 모델이 본 문장을 분류기도 외운 혼입이 생긴다.
이를 막기 위해:

> 최종 통합셋에서 split하되 **분류기는 모델 train split 페어로만 학습**하고,
> **델타는 모델·분류기 모두에게 held-out인 공통 test split predictions에서만** 측정한다.

데이터·모델·분류기 모두 우리 통제 하에 있으므로(외부 고정 모델 아님) split 정렬은
우리가 자유롭게 강제할 수 있다.

### 5. 보류 — 구성 타당성(OOD) 점검

분류기가 *일반적 구어성*을 잡는지, *SBC 표면 아티팩트*(전사 관습·특정 토큰)를
암기했는지의 구분(다른 코퍼스 OOD 정확도 점검)은 **현 단계에서 보류**한다. §4의
split 정렬로 "외운 문장 혼입"은 제거되지만, "SBC 표면 흉내 vs 진짜 구어성"의 잔여
모호성은 남는다. 보고서 방어선 판단상 지금은 불필요하다고 결정 — 추후 재검토.
(운영 메모는 `status.md` 보류 항목.)

## Consequences

### 긍정적

- style_separation(상대 분기)·evaluate(분포 정렬)가 못 보던 **절대·방향 구어성 축**을
  단일 수치로 제공 → "얼마나 구어스러워졌는가"에 직접 답.
- 피처 대부분 기존 추출기 재사용(filler/POS/분절) — 신규는 `pronoun_ratio`·
  `contraction_rate` 둘. scikit-learn 외 의존성 추가 없음.
- `style_separation.py`가 열어둔 "분류기 기반 separability" 자리를 실제로 채움.

### 부정적 / 리스크

- 분리도가 modest(auc~0.75, 천장은 피처 신호량이 정함) — 우리 formal이 LLM 역번역이라
  두 클래스가 본디 가깝기 때문. 보고서는 이 한계를 명시한다.
- 영속화된 model+scaler 아티팩트 관리 필요(학습 split 의존 → split 바뀌면 재학습).
- `predict_proba`는 근사 — 절대 확률값보다 *델타의 부호·크기* 해석에 무게.

## Update (2026-06-03, 실측 반영)

구현 중 실측에서 두 가지가 드러나 §2·§3을 위와 같이 개정했다(상세 수치·재현:
`.work/note-spokenness-classifier-findings.md`):

1. **팀원 88%는 pause 주석 아티팩트였다.** pause 토큰 그대로면 RF acc 0.81이지만
   strip하면 0.47(chance). 분류기가 구어성이 아니라 "`<pause:*>` 존재"를 학습한 것.
   → pause 제외, filler 포함(§3). 이는 §5에서 보류했던 구성 타당성 우려가 OOD가
   아니라 in-distribution에서 더 치명적으로 드러난 형태다.
2. **"더 잘 학습"으론 천장을 못 넘는다.** RF 용량↑(0.709→0.712)은 무효, 모델 교체
   (LogReg 0.743 / GB 0.749)만 modest 개선 → §2를 LogReg로. 현실적 천장 auc~0.75.

§5(OOD 보류)는 유지하되, 위 발견으로 "보고서에 한계 명시"가 선택이 아닌 전제가 됨.

## References

- [ADR-0013](0013-semi-formal-data-sourcing.md) §6 — 스타일 분리도 메트릭(상대 분기)
- [ADR-0005](0005-style-as-dataset-metadata.md) — 스타일을 데이터셋 메타속성으로
- `scripttuner/training/style_separation.py` — 분류기 기반 separability 보류 메모
- `scripttuner/preprocessing/stats.py` `_pos_stats` — 재사용할 spaCy 피처 추출기
- `.work/teammate/구어체문어체분류.ipynb` — 팀원 시제작(SVM·시각화 별도 기여)