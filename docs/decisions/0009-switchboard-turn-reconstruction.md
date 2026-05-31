# ADR-0009: Switchboard 턴 재구성 — 타임스탬프 인터리브로 ③ 재사용

- **Status**: Accepted
- **Date**: 2026-05-26

## Context

보조 구어 코퍼스로 Switchboard(OpenSLR #5, MSU transcripts)를 도입한다 (라이선스는 [ADR-0010](0010-switchboard-license-policy.md)). 이 코퍼스는 SBCSAE(CHAT)와 턴 구조 표현이 근본적으로 다르다.

- 한 대화의 두 화자가 **별도 파일**에 있다: `swNNNNA-ms98-a-trans.text`, `swNNNNB-ms98-a-trans.text`.
- 각 면(side) 파일은 그 화자의 발화 세그먼트 + **상대가 말하는 구간을 `[silence]`로 채운 라인**으로 구성된다. PoC 측정상 전체 라인의 **36.5%가 `[silence]`/비언어**다.
- 각 라인은 `<utt_id> <start_sec> <end_sec> <text>` 형식으로 **양면 모두 정밀 타임스탬프**를 갖는다(강제정렬 기반).

반면 ③ `monologue.py`의 `build_monologues`는 **단일 스트림에서 화자가 교대(alternate)** 한다는 전제로 동작한다 (`monologue.py:118-143`): 같은 화자면 버퍼에 append, 다른 화자면 백채널이면 skip([ADR-0004](0004-backchannel-handling.md)), 실발화면 flush. 따라서:

- 한 면만 ③에 넘기면 모든 utterance가 동일 화자 → **전부 한 덩어리로 병합**되고 ADR-0004 백채널 skip이 한 번도 작동하지 않는다.
- 빈 텍스트(`[silence]` 유래) utterance가 ③에 도달하면 `is_backchannel("") == False`라, 다른 화자의 침묵이 **버퍼를 조기 flush**시켜 모놀로그를 파편화한다.

## Decision

Switchboard 어댑터는 A/B 두 파일을 **`t_start_ms` 기준 단일 시간순 스트림으로 인터리브**하여 SBCSAE와 동일한 "화자 교대 스트림"을 복원한다. 그 결과 ③ `monologue.py`와 ADR-0004는 **변경 없이 재사용**된다.

구체 계약:

1. **파싱 단위 = 대화 1건.** `swNNNNA` + `swNNNNB` 두 파일을 함께 읽어 stem `swNNNN`으로 출력. 인터리브는 **Switchboard 파서의 책임**(ADR-0006: 어댑터가 포맷 특수성 흡수, ③~⑤ 불변).
2. **정렬 키 = `t_start_ms`.** 안정 정렬로 동률 시 결정적 순서 보장(읽기 순서 A→B 유지).
3. **`[silence]`·비언어 마커는 전부 drop, `<pause>` 합성하지 않는다.** ADR-0003의 `<pause:*>`는 CHAT 전사자가 **사람이 판단해 찍은** 마커(`(.)`/`(..)`)에서 온다. Switchboard `[silence]`는 **강제정렬이 만든 기계적 dead air**라 성격이 다르며, 같은 `<pause>` 토큰으로 합치면 ⑤ stats(`pause_*_per_pair`)와 학습 신호에서 두 코퍼스가 의미가 다른 토큰을 한 이름으로 섞게 된다(cross-corpus feature 오염). "양쪽 동시 침묵 → 발화 내 휴지" 구분은 교차 면 interval-overlap + duration 임계값 비용 대비 이점이 낮고(이미 `um/uh/`재시작 등 disfluency 신호가 풍부), 의미가 모호하여 채택하지 않는다.
4. **마커 정규화 후 빈 텍스트 utterance는 ③ 도달 전 drop**(필수 — 위 조기 flush 방지).
5. **백채널 사전은 어댑터가 `build_monologues(backchannel_words=...)`로 주입**한다(코퍼스 격리, ③ 불변). 단 `_word_tokens`가 하이픈을 분할하므로(`um-hum` → `["um","hum"]`, `uh-huh` → `["uh","huh"]`), 사전에는 통짜 `um-hum`이 아니라 **분리 컴포넌트**(`um, hum, uh, huh, ...`)를 넣어야 매칭된다.

## Consequences

### 긍정적

- ③ `monologue.py`와 ADR-0004를 한 줄도 고치지 않고 재사용 → ADR-0006의 "③~⑤ 코퍼스 무관" 계약 유지.
- 백채널 vs 실턴 구분이 B의 **실제 발화 내용** 기준으로 이뤄진다(원칙적). 방식 A의 침묵-길이 휴리스틱보다 정확.
- PoC 실측상 ≥30토큰 monologue run 약 29.7k개 — SBCSAE 최종 페어(1,757) 대비 압도적 증분 가능.

### 부정적

- 파서가 "파일 1개 → utterances" 대신 "대화(2파일) → utterances"가 되어 CHAT 어댑터보다 파싱 단위가 크다. run/CLI 오케스트레이션이 A/B 페어링을 해야 함 — **별도 결정으로 파킹**.
- 동시발화(overlap)의 시간순 선형화는 근사다. 단 `[silence]` 구조상 overlap은 대부분 백채널 케이스라 skip 로직이 흡수한다.
- 백채널 사전·`min_tokens` 튜닝이 Switchboard 특성에 맞게 필요.

## Alternatives Considered

| 안 | 기각 이유 |
|---|---|
| **방식 A**: 면별 `[silence]`-구분 run을 monologue 후보 | ③ 재사용 불가(전부 동일 화자 → 한 덩어리). 침묵-길이 임계값이라는 신규 휴리스틱 필요. "A의 머뭇거림"과 "B의 긴 턴"을 A 파일만으로는 구분 불가 → ADR-0004가 주던 백채널/실턴 판별을 상실 |
| ③를 per-side 입력 지원하도록 수정 | 코퍼스별 분기를 공통 모듈에 주입 → ADR-0006 위반 |
| `[silence]`를 `<pause>`로 매핑 | Decision 3 참조 — cross-corpus pause feature 오염 |

## References

- [docs/design/preprocessing_pipeline.md](../design/preprocessing_pipeline.md)
- [ADR-0003](0003-pause-marker-tokenization.md), [ADR-0004](0004-backchannel-handling.md), [ADR-0006](0006-adapter-structure-and-common-ir.md)
- [ADR-0010](0010-switchboard-license-policy.md) — Switchboard 라이선스
- `scripttuner/preprocessing/monologue.py` — 재사용 대상 ③ 모듈
