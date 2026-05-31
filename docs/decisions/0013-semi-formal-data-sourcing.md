# ADR-0013: Semi-formal 스타일 데이터 조달 방안

- **Status**: Accepted
- **Date**: 2026-05-30

## Context

[ADR-0005](0005-style-as-dataset-metadata.md)에서 스타일을 데이터셋 단위 메타속성으로
부여하기로 하고, `semi_formal`은 "별도 코퍼스 확보 필요"로 남겨두었다.
`status.md` 다음 액션에서도 조달 방안("teacher-LLM 합성 vs 외부 코퍼스")이 미결이었다.
이 ADR은 semi_formal의 **필요성·정의·소스 품질·출처**를 재검토해 그 미결을 닫는다.

코드/설계상 슬롯은 이미 예약되어 있다: `training/style.py`의 `semi_formal` spec,
formatter의 `style_token`(`<style:semi_formal>`), control-token 분기.

## Decision

### 1. 범위 — semi_formal은 제품 핵심이 아니라 제안서 시연용 옵션 축

casual 단일로 프로젝트 핵심(문어체→자연 구어체 변환)은 이미 증명되었다(마일스톤 ✅).
semi_formal은 제안서가 약속한 "발화 스타일 조건부 생성(두 스타일 제어 분기)"을
시연하기 위한 옵션 축이다. 핵심 가치 검증의 전제 조건이 아니다.

### 2. 정의 — 즉흥 발화 + 격식 절제

`semi_formal` = **즉흥성은 casual과 공유**하되 **register만 절제**한 발화
(필러 절제, 담화 마커 `well`/`actually`/`to be fair` 위주, 어휘·구성 정돈).
casual과의 차이는 *즉흥성 여부가 아니라 격식 수준*이다.

### 3. 소스 품질 필터 — scripted/rehearsed(낭독체) 소스 금지 (casual·semi_formal 공통)

프로젝트 목적은 OPIc 답변의 "대본 읽는 느낌" 제거다. 따라서 리허설된 낭독체
(TED 강연 등)는 학습 *목표*와 정면충돌한다 — 거기서 필러가 적은 것은 *격식*이
아니라 *준비됨* 때문이며, 이를 학습하면 모델이 "정돈된 낭독 폴리시"를 익혀
제거 대상을 오히려 생성한다. **TED 등 scripted 소스는 양 스타일 모두에서 배제한다.**

### 4. Path B(LLM이 output/target 합성) 기각

이 파이프라인에서 LLM은 **input(`formal_text`)만 생성**한다(실제 발화의 역번역).
**output(`spoken_text`)은 항상 실제 코퍼스 발화여야 한다** — 그것이 본 모델의 존재
이유다. target을 LLM으로 합성하면, 제안서가 경쟁 상대로 지목한 바로 그 약점
("상용 LLM의 문어체 편향·기계적 필러 삽입")을 학습 타깃에 주입하게 된다.
→ semi_formal의 output 합성 경로는 채택하지 않는다. 출처는 실제 발화 코퍼스뿐.

### 5. 출처 — 실제 즉흥+격식절제 spoken 코퍼스

| 우선 | 코퍼스 | 라이선스 | 포맷/비고 |
|---|---|---|---|
| 1 | **AMI + ICSI Meeting** | CC BY 4.0 | NXT XML — 어댑터 1개로 둘 다 획득. 다자 회의 → 동일화자 turn 재조립([ADR-0009](0009-switchboard-turn-reconstruction.md) 패턴 재사용) |
| 2 | **MICASE** (학술구어) | CC BY 4.0 *(현행 확인 필요)* | 자체 XML. 강의=모놀로그라 OPIc register 적합도 우수 |

제외: People's Speech·GigaSpeech(대용량이나 register 통제 불가 → 큐레이션 과다),
CABNC·CallFriend(casual), SCOTUS(법정/법률 도메인), SLx(vernacular 유도).

참고로 semi_formal 후보(CC BY 4.0)는 메인 casual 코퍼스 SBCSAE(CC BY-ND,
[ADR-0002](0002-sbcsae-license-policy.md))보다 **라이선스가 더 자유롭다**.

### 6. 검증 선행 — 스타일 분리도 메트릭

착수 전, casual vs semi_formal 출력이 *실제로 분기*되는지 측정할 메트릭을 정의한다.
현 spoken-ness 메트릭은 ref 정렬도만 측정할 뿐 스타일 간 분리는 측정하지 않으므로,
이대로 학습하면 두 스타일이 collapse해도 감지할 수 없다.

## Consequences

### 긍정적

- 라이선스가 casual 데이터보다 깨끗(CC BY 4.0) — 산출물 공유 제약 없음
- 어댑터 1개(NXT)로 AMI+ICSI 동시 확보, 모놀로그 재조립 패턴 재사용 → 난이도 MEDIUM
- scripted 금지·Path B 기각을 명문화 → 향후 데이터 추가 시 선별 기준 고정

### 부정적 / 리스크

- 새 포맷(NXT) 어댑터 신규 작업 필요(MEDIUM)
- AMI의 약 ⅔는 scenario(리모컨 설계) 기반이라 주제 협소 — 즉흥 발화이나 어휘 다양성 한계
- 스타일 분리도 메트릭 신규 정의 필요
- MICASE는 라이선스 현행(CC BY 4.0) 확인 전까지 2순위 보류

## References

- [ADR-0005](0005-style-as-dataset-metadata.md) — 스타일을 데이터셋 메타속성으로
- [ADR-0009](0009-switchboard-turn-reconstruction.md) — 다자 발화 모놀로그 재조립 패턴
- [ADR-0011](0011-corpus-adapter-interface.md) — 코퍼스 어댑터 인터페이스
- [docs/design/dataset_review.md](../design/dataset_review.md) — 데이터 확보 로드맵
- [background/QAIP_proposal_report.md](../../background/QAIP_proposal_report.md) — 발화 스타일 조건부 생성