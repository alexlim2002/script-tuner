# 프로젝트 진행 현황

> 본 문서는 **상태 추적용**이다. 일상적으로 업데이트한다. 정적 설계는 `docs/design/`, 결정 이력은 `docs/decisions/`(ADR), 거친 작업 메모는 `.work/`(gitignore)에 둔다.

마지막 업데이트: 2026-05-30

---

## 마일스톤 (단계)

| 단계 | 내용 | 상태 |
|---|---|---|
| 전처리 파이프라인 | 어댑터 구조 + ①~⑤(파서·cleaner·monologue·pairs·stats) + CLI | ✅ |
| 데이터 확보·보강 | SBCSAE 1,757 pairs · Switchboard 34,895 monologues(LLM 전) · Semi-formal: AMI+ICSI 조달 결정([ADR-0013](decisions/0013-semi-formal-data-sourcing.md)), 미착수 | ⏳ |
| 진단 모듈 | 구어성 진단 (⑤ stats feature를 ground truth로) | 예정 |
| 변환 모델 학습 | T5Gemma 2-1B LoRA(seq2seq, bf16, 8GB GPU) 학습·추론·평가·시각화 파이프라인 완성. casual 단일 스타일, ref-정렬된 spoken-ness 메트릭 확인. semi_formal 미착수 | ✅ (casual) |
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

1. 보고서/발표 준비 — 집계 산출물은 `report/`에 정리됨([metrics_summary.md](../report/metrics_summary.md) = 1-epoch vs 조기종료 비교, training_curves.png, samples.md). 실제 보고서·발표 문서 작성 남음.
2. Semi-formal 두 번째 학습 사이클 (조달 방안 결정됨, cf. [ADR-0013](decisions/0013-semi-formal-data-sourcing.md)) — 착수 전 선행: ⓐ casual vs semi_formal **스타일 분리도 메트릭** 정의, ⓑ AMI+ICSI **NXT 어댑터**(ADR-0011 인터페이스) 구현, ⓒ 모놀로그 재조립(ADR-0009 패턴) → ④ pairs → split/format → 재학습. (MICASE는 라이선스 현행 확인 후 2차 보강)
3. (선택) Switchboard ④ LLM pairs → SBCSAE와 `aggregate` 합산해 학습 데이터 확장 (비용 발생).
4. 학습 target 구두점 정규화 (모델 피드백 대응, cf. ADR-0012):
   - **Tier A (코드 완료)** — (a) `' .'→'.'`·중복 종결 정리(`text_normalize.normalize_punctuation`, ④ target + generate 출력), (b) cleaner 마커 제거 보강(`[% laugh]`/`&{n=…}`/`XX`/밑줄). 검증: `' .'` 23,698→0, 마커 raw 616→0(X-rays 7 보존). **pairs 재생성 → 재학습 대기**.
   - **Tier B (PoC 통과 → 본 적용 대기)** — 쉼표·`?` 재구두점(방식 A: 자유생성+토큰검증, 깨지면 Tier A 폴백). 59쌍 PoC: 실제 단어변경 ~14%(전부 폴백), 품질 양호, `!`는 미채택. 본 구현 남음: 전량 재구두점 → pause 인덱스 재삽입 → pairs/split/formatted 재생성 → 1회 재학습. LLM은 OpenRouter 1차 / Groq 2차.
   - 문법 교정(피드백 3·4)은 스코프 밖 — "입력은 문법적 정상" 전제, L2 강건성은 보류 항목.

## 보류 / 추후 결정

- ~~**Semi-formal 스타일 데이터 확보 방안**~~ → 결정됨([ADR-0013](decisions/0013-semi-formal-data-sourcing.md)): AMI+ICSI(CC BY 4.0) 1순위, MICASE 2순위. TED 등 scripted 소스·LLM target 합성은 배제.
- **제어 토큰 학습 전략** — Semi-formal 데이터 확보 후. 제어 토큰 슬롯(casual / semi_formal)은 formatter에 예약됨
- **진단 모듈 feature set 최종화** — ⑤ stats 산출 결과 보고 결정
- **모델 escalation** — 현 T5Gemma 2-1B 결과가 ref와 잘 정렬. 품질 부족 시 T5Gemma 2-4B 또는 Gemma 4 escalation (12GB+ GPU 필요)
- **few-shot 도입 시점** — 현재 zero-shot 결과 양호. 후속 코퍼스 추가/품질 이슈 시 재검토
- **정량 품질 메트릭** — 현 spoken-ness 메트릭(filler/pause/length/lexical density) 외에 BLEU/embedding similarity 도입 여부
