# 프로젝트 진행 현황

> 본 문서는 **상태 추적용**이다. 일상적으로 업데이트한다. 정적 설계는 `docs/design/`, 결정 이력은 `docs/decisions/`(ADR), 거친 작업 메모는 `.work/`(gitignore)에 둔다.

마지막 업데이트: 2026-05-31

---

## 마일스톤 (단계)

| 단계 | 내용 | 상태 |
|---|---|---|
| 전처리 파이프라인 | 어댑터 구조 + ①~⑤(파서·cleaner·monologue·pairs·stats) + CLI | ✅ |
| 데이터 확보·보강 | SBCSAE 1,757 pairs · Switchboard 34,895 monologues(LLM 전) · Semi-formal AMI 2,055 pairs([ADR-0013](decisions/0013-semi-formal-data-sourcing.md)) | ✅ |
| 진단 모듈 | 구어성 진단 (⑤ stats feature를 ground truth로) | 예정 |
| 변환 모델 학습 | T5Gemma 2-1B LoRA(seq2seq, bf16, 8GB GPU) 학습·추론·평가·시각화 파이프라인 완성. casual+semi_formal 통합 학습으로 제어 토큰 스타일 분기 확인(paired generation 분리 검증). | ✅ |
| 백엔드 / UI | 서빙 · 사용자 인터페이스 | 예정 |

세부 진행 이력은 `git log` + 아래 ADR 목록 참조. 파인튜닝 사전 준비(split/format) 흐름은 [`docs/design/finetuning_pipeline.md`](design/finetuning_pipeline.md) (상세 한국어판: [`finetuning_pipeline_ko.md`](design/finetuning_pipeline_ko.md)) 참조.

## 결정 이력 (ADR)

- [ADR-0001](decisions/0001-jsonl-output-format.md) — 학습 데이터 출력 포맷으로 JSONL 채택
- [ADR-0002](decisions/0002-sbcsae-license-policy.md) — SBCSAE 라이선스 대응 (다운로드 스크립트 + gitignore)
- [ADR-0003](decisions/0003-pause-marker-tokenization.md) — 포즈 마커 특수 토큰화
- [ADR-0004](decisions/0004-backchannel-handling.md) — 백채널 처리 정책
- [ADR-0005](decisions/0005-style-as-dataset-metadata.md) — 스타일 레이블을 데이터셋 메타속성으로
- [ADR-0006](decisions/0006-adapter-structure-and-common-ir.md) — 어댑터 구조 + 공통 IR
- [ADR-0007](decisions/0007-llm-client-provider-agnostic-and-caching.md) — LLM 클라이언트 provider-agnostic + 디스크 캐싱
- [ADR-0008](decisions/0008-pause-token-strip-on-llm-input.md) — LLM 입력 전 pause 토큰 strip (spoken 보존)
- [ADR-0009](decisions/0009-switchboard-turn-reconstruction.md) — Switchboard 턴 재구성: 타임스탬프 인터리브로 ③ 재사용
- [ADR-0010](decisions/0010-switchboard-license-policy.md) — Switchboard(MSU transcripts) 라이선스 정책
- [ADR-0011](decisions/0011-corpus-adapter-interface.md) — 코퍼스 어댑터 인터페이스 + stem-centric 파이프라인 (`run --through`)
- [ADR-0012](decisions/0012-target-punctuation-normalization.md) — 학습 target 구두점 정규화 (Tier A 즉시 / Tier B 재구두점 / 문법 스코프 아웃)
- [ADR-0013](decisions/0013-semi-formal-data-sourcing.md) — Semi-formal 데이터 조달 (scripted 금지 · Path B 기각 · AMI+ICSI 1순위 / MICASE 2순위)

## 다음 액션 (단기)

1. 보고서/발표 준비 — 집계 산출물은 `report/`에 정리됨([metrics_summary.md](../report/metrics_summary.md) = repunct before/after + 학습길이 방법론 노트, training_curves.png, [samples.md](../report/samples.md) = es vs repunct 정성 쌍). 실제 보고서·발표 문서 작성 남음.
2. Semi-formal 두 번째 학습 사이클 (조달 방안 결정됨, cf. [ADR-0013](decisions/0013-semi-formal-data-sourcing.md)) — 데이터 준비까지 완료, **전량 pairs→학습이 남음**:
   - ⓐ 스타일 분리도 메트릭 ✅(`training/style_separation.py`, Cohen's d·collapse 판정; pause/tokens는 판정 제외=코퍼스 아티팩트). 핵심 분리 축은 **문장 분절(`tokens_per_sentence`)** — casual은 짧게 토막, semi_formal은 절을 길게 이어붙임. filler/lexical/phrasal만으론 이 발화 리듬을 못 잡아 collapse 오판이 나므로 verdict에 분절 포함. 판정은 monologue 원문이 아닌 *동일 입력 paired 생성* 출력으로 한다(표기관습 교란 배제).
   - ⓑ AMI NXT 어댑터+다운로더+cleaner ✅(`download/run ami`, 171회의·8,920 monologue·636k토큰; segments 단위·speaker=global_name·철자표기 `T_V_→TV` 복원). ADR-0012(`word .`/쉼표)는 AMI(NXT)엔 비해당 — 표기관습이 달라 이미 깨끗.
   - ⓒ 모놀로그 재조립 ✅(monologue.py 무변경 재사용).
   - ⓓ 샘플러 ✅(`scripttuner sample ami`, casual과 스케일매치 ~2,055; scenario:nonscenario=1:1, 화자단위 누수방지; `data/monologues_sampled/`).
   - ⓔ pairs PoC ✅(7쌍, `--style semi_formal`, 동일모델 gpt-oss-120b:free, 품질·비용≈0 확인).
   - ⓕ `pairs --all --monologues-subdir` + rate-limit 재시도 일반화 ✅(`llm/rate_limit.py`의 `RateLimitRetryClient`를 모든 provider에 적용; max_retries 반복). + OpenRouter provider 라우팅(`LLM_PROVIDER_SORT=price`, `MAX_PROMPT/COMPLETION`)으로 저가 provider 한정 — Groq 무료 TPM 8k가 너무 낮아 OpenRouter 유료 저가(prompt≤0.2/compl≤0.4 $/M, gpt-oss-120b)로 전환.
   - ⓖ 전량 pairs ✅(**2,055/114 stem, skip 0, 429 0**) → aggregate ✅(2,055) → split ✅(train 1,636/val 209/test 210, 화자 47명, 누수 0) → format t5gemma2-1b ✅(styles=[semi_formal]).
   - ⓗ 통합(casual+semi_formal) 학습·판정 ✅ — COMBINED 데이터셋(train 3,049/val 382/test 381, 누수 0) Kaggle 학습 → 어댑터 `aip-scripttuner-team/scripttuner-t5gemma2-1b-combined`. **paired generation으로 분리 확인**(collapse 아님): 제어 토큰이 두 스타일을 분기시킨다 — semi_formal이 절을 길게 이어붙이고(분절 d>0.2, 데이터·출력 모두 기대방향 일치) casual은 pause+짧은 토막. ※ filler_rate는 데이터부터 기대 반대(AMI 회의의 머뭇거림=장르 차이)라 register 신호로 보지 않음.
   - **남음**: 결과 report 정리. ICSI(동일 NXT)는 필요시 보강. MICASE는 라이선스 현행 확인 후 2차 보강. ※ AMI 다운로더는 공식 v1.6.2(CC BY 4.0); OpenSLR 미러 v1.6.1은 구 라이선스(CC BY-NC-SA 2.5) 동봉.
3. (선택) Switchboard ④ LLM pairs → SBCSAE와 `aggregate` 합산해 학습 데이터 확장 (비용 발생).
4. 학습 target 구두점 정규화 (모델 피드백 대응, cf. ADR-0012):
   - **Tier A (코드 완료)** — (a) `' .'→'.'`·중복 종결 정리(`text_normalize.normalize_punctuation`, ④ target + generate 출력), (b) cleaner 마커 제거 보강(`[% laugh]`/`&{n=…}`/`XX`/밑줄). 검증: `' .'` 23,698→0, 마커 raw 616→0(X-rays 7 보존). **pairs 재생성 → 재학습 대기**.
   - **Tier B (완료·검증됨 ✅)** — 쉼표·`?` 재구두점(방식 A: 자유생성+토큰검증, 깨지면 Tier A 폴백, `!` 미채택). 전량 재구두점 → pause 재삽입 → pairs/split/formatted 재생성 → Kaggle T4에서 재학습(`t5gemma2-1b-SBCSAE-lora-repunct`, 어댑터 HF private push). **결과: 쉼표 0→4.35/item(ref 3.73), `' .'` 4.41→0, 구어성(filler/pause/length) 회귀 없음** — `-es`(동일 8ep, 데이터만 다름)와 단일변수 비교. 상세 [report/metrics_summary.md](../report/metrics_summary.md). LLM은 OpenRouter 1차 / Groq 2차.
   - 문법 교정(피드백 3·4)은 스코프 밖 — "입력은 문법적 정상" 전제, L2 강건성은 보류 항목.

## 보류 / 추후 결정

- ~~**Semi-formal 스타일 데이터 확보 방안**~~ → 결정됨([ADR-0013](decisions/0013-semi-formal-data-sourcing.md)): AMI+ICSI(CC BY 4.0) 1순위, MICASE 2순위. TED 등 scripted 소스·LLM target 합성은 배제.
- ~~**제어 토큰 학습 전략**~~ → 통합 학습으로 검증됨(ⓗ). 제어 토큰이 두 스타일을 분기시킴(분절 축)
- **진단 모듈 feature set 최종화** — ⑤ stats 산출 결과 보고 결정
- **모델 escalation** — 현 T5Gemma 2-1B 결과가 ref와 잘 정렬. 품질 부족 시 T5Gemma 2-4B 또는 Gemma 4 escalation (12GB+ GPU 필요)
- **few-shot 도입 시점** — 현재 zero-shot 결과 양호. 후속 코퍼스 추가/품질 이슈 시 재검토
- **정량 품질 메트릭** — 현 spoken-ness 메트릭(filler/pause/length/lexical density) 외에 BLEU/embedding similarity 도입 여부
