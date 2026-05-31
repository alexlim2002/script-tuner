# 데이터셋 검토: SBCSAE

> 본 문서는 본 프로젝트가 활용할 1차 데이터 소스인 SBCSAE의 특성·강점·약점·활용 방안을 정리한 정적 설계 문서이다. 진행 상황은 [`status.md`](../status.md)를, 결정 이력은 [`decisions/`](../decisions/)를 참고한다.

## 개요

| 항목 | 값 |
|---|---|
| 데이터셋 | Santa Barbara Corpus of Spoken American English (SBCSAE) |
| 구성 | 60개 자연 대화 전사, 약 25만 단어 |
| 포맷 | CHAT (CHILDES) |
| 라이선스 | CC BY-ND 3.0 US (cf. [ADR-0002](../decisions/0002-sbcsae-license-policy.md)) |
| 1차 검토 대상 | SBC016.cha — 산타바바라 오디오 매장 판매 대화 1건 (4 화자, 약 22분, 1,534 발화 라인) |

## 적합성 평가

### 강점

- **순수 원어민 즉흥 발화**: `um`, `uh`, `&=tsk`, `&=laugh`, `you know`, `I mean`, `well`, `gonna`, `kinda` 등 필러·축약·담화 마커가 자연 분포한다. 제안서가 구어체 학습 신호로 명시한 요소들이 그대로 들어 있다.
- **격식 수준이 OPIc과 가까움**: 친밀 잡담이 아닌 판매원-고객의 반(半)정중 대화. **Casual / Semi-formal spoken 양쪽 추출 가능성**이 있다.
- **풍부한 메타 어노테이션**: 필러·웃음·발화 끊김(`+/.`, `+...`)·오버랩(`⌈ ⌉`)·자기수정이 명시적으로 태깅되어, 구어체 요소의 삽입/제거 위치를 학습 신호로 쓸 수 있다.
- **OPIc 친화 토픽**: 쇼핑/구매 결정은 OPIc 빈출 주제.

### 약점

1. **절대량 부족** — 60개 대화·25만 단어는 LoRA 파인튜닝 기준 작은 편. 보조 코퍼스 필수.
2. **CHA 포맷 전처리 부담** — 타임스탬프, 오버랩 마커, 비언어 태그, 발화 끊김, 특수 발음 표기 등 처리 항목이 많다 (cf. [`preprocessing_pipeline.md`](preprocessing_pipeline.md)).
3. **턴 구조 미스매치** — SBCSAE는 짧은 turn + 잦은 백채널 + 오버랩 중심. OPIc은 1인 2~3분 모놀로그. 그대로 학습 시 모델이 짧게 끊는 응답을 생성할 위험. 동일 화자의 연속 turn을 monologue로 재조립하는 후처리 필요 (cf. [ADR-0004](../decisions/0004-backchannel-handling.md)).
4. **주제 다양성 한계** — 한 파일당 한 주제. 다양한 OPIc 토픽 커버 불가.

## 활용 방향

> SBCSAE는 본 학습 데이터의 **메인 구어 코퍼스**로 사용하되, 단독으로는 부족하다. 보조 코퍼스 및 monologue 코퍼스로 보강한다.

### 데이터 확보 로드맵

| 우선순위 | 데이터 | 용도 | 비고 |
|---|---|---|---|
| 1 | SBCSAE 60개 전체 | 메인 구어 코퍼스 (Casual) | TalkBank/OpenSLR, CHA 포맷 동일 → 파이프라인 재사용 |
| 2 | Switchboard / CallHome / BNC Spoken | 대규모 구어 보강 | 도메인 다양성 확보 |
| 3 | AMI + ICSI Meeting (CC BY 4.0), 차순위 MICASE | **Semi-formal** monologue 보강 (cf. [ADR-0013](../decisions/0013-semi-formal-data-sourcing.md)) | 즉흥+격식절제 회의/학술구어. **scripted(TED 등) 배제** — "대본 읽는 느낌"은 제거 대상이지 학습 목표가 아님 |
| 4 | 상용 LLM 활용 역변환 | (문어체, 구어체) 페어 생성 | 제안서 명시 방식 |

### 스타일 레이블링

SBCSAE는 데이터셋 단위로 `style: "casual"`로 일괄 레이블한다. 청크별 분류기는 두지 않는다. 자세한 근거는 [ADR-0005](../decisions/0005-style-as-dataset-metadata.md).

## 참고: SBC016.cha 특징

- 화자 4명 등록(TAMM, BRAD, TODD, JONA) — 실제 발화는 TAMM/BRAD 중심
- 약 22분 대화, 1,534 발화 라인
- 주제: 카세트 데크 구매 상담
- PoC 검증용으로 적합 (적당한 길이, 풍부한 마커, 한정된 화자)