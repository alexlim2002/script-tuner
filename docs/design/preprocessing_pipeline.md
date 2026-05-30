# 전처리 파이프라인 설계

> 본 문서는 CHA 원본을 (문어체, 구어체) 병렬 코퍼스로 변환하는 전처리 파이프라인의 정적 설계서이다. 본 단계에서는 학습/추론은 다루지 않는다. 관련 결정 이력은 [`../decisions/`](../decisions/), 진행 상황은 [`../status.md`](../status.md)를 참고한다.

## 목표

CHA(CHILDES) 포맷의 원어민 대화 전사 데이터를 **OPIc 모놀로그 학습용 (문어체 ↔ 구어체) 병렬 코퍼스**로 변환한다.

## 입출력

- **입력**: 코퍼스별 원본 파일 (현재는 CHAT `.cha`, 향후 NXT/XML/plain 등)
- **출력**: 모듈별 JSONL 산출물 (cf. [ADR-0001](../decisions/0001-jsonl-output-format.md))

## 아키텍처

본 파이프라인은 **어댑터 구조 + 공통 IR** 패턴을 따른다 (cf. [ADR-0006](../decisions/0006-adapter-structure-and-common-ir.md)).

- ① 파서 + ② Cleaner는 **코퍼스/포맷별 어댑터**에 묶임
- 어댑터는 자기 포맷을 공통 IR(`Utterance`)로 변환
- ③ Monologue 재조립, ④ LLM 역변환, ⑤ 통계는 IR만 다루므로 **코퍼스 무관**

### 패키지 구조

```
scripttuner/
├── preprocessing/
│   ├── ir.py                # 공통 IR — Utterance, Monologue, Pair dataclass
│   ├── chat/                # CHAT (CHILDES) 어댑터 — SBCSAE 등
│   │   ├── parser.py        # ① CHAT 파서
│   │   └── cleaner.py       # ② CHAT 정규화
│   ├── switchboard/         # (미래) Switchboard 어댑터
│   ├── monologue.py         # ③ 공통
│   ├── pairs.py             # ④ 공통 — LLMClient Protocol + convert_to_formal
│   └── stats.py             # ⑤ 공통
├── persistence/             # 디스크 적재/직렬화 영역
│   ├── jsonl.py             #   JSONL I/O (dataclass ↔ dict)
│   └── cache.py             #   sha256-keyed JSON KV 캐시 (LLM 응답 등)
└── llm/                     # LLM 클라이언트 (provider-agnostic, cf. ADR-0007)
    └── openai_compatible.py #   OpenAI SDK 래퍼
```

## 파이프라인 구조

```
[입력] *.cha (or other corpus format)
   │
   ▼
① 파서 (e.g. chat/parser.py)
   - 코퍼스별 헤더/메타 파싱
   - 발화 라인 → Utterance(speaker, text=raw, t_start, t_end, metadata)
   - 멀티라인 발화 결합
   - 마커는 손대지 않음 (text에 그대로, 타임스탬프 토큰만 분리)
   │
   ▼ parsed/*.jsonl
   │
② Cleaner (e.g. chat/cleaner.py)
   - 코퍼스별 마커 처리 (CHAT의 경우 아래 "마커 처리 정책" 참조)
   - text를 정규화된 형태로 갱신
   │
   ▼ cleaned/*.jsonl
   │
③ Monologue 재조립 (monologue.py) — 코퍼스 무관
   - 동일 화자 연속 발화 병합
   - 백채널(Okay/Yeah/Mhm 등 짧은 응답) 식별 후 skip (cf. ADR-0004)
   - 최소 길이 필터링 (예: ≥30 tokens)
   │
   ▼ monologues/*.jsonl
   │
④ (구어체 → 문어체) 역변환 (pairs.py) — 코퍼스 무관
   - LLM 호출로 정제된 문어체 페어 생성 (provider-agnostic, cf. ADR-0007)
   - 입력 전처리: <pause:*> 토큰 strip (LLM은 토큰을 모름, cf. ADR-0008)
   - 출력 후처리: typography ASCII 정규화 (스마트 따옴표·em-dash → ASCII)
   - 디스크 캐싱: sha256(prompt_version+model+stripped_input) 키 (cf. ADR-0007)
   - 메타: style="casual" (cf. ADR-0005)
   │
   ▼ pairs/*.jsonl
   │
⑤ 통계 / 검증 (stats.py) — 코퍼스 무관
   - 카운트, 길이 분포, 사전 기반(filler/pause), POS 기반(어휘 밀도/구동사 비율)
   - 입력: pair JSONL  출력: 파일당 단일 JSON
   - 진단 모듈의 ground truth로 활용
   │
   ▼ stats/*.json
```

## 공통 IR (`Utterance`)

모든 어댑터의 출력 표준. 코퍼스 무관 최소 공통 필드만 둔다.

```python
@dataclass(frozen=True)
class Utterance:
    source: str                       # "SBCSAE", "Switchboard" 등
    utterance_id: str                 # 코퍼스 내 고유 ID
    speaker: str                      # 필수 — monologue 재조립의 키
    text: str                         # stage에 따라 raw or cleaned
    t_start_ms: int | None = None
    t_end_ms: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

자세한 설계 근거는 [ADR-0006](../decisions/0006-adapter-structure-and-common-ir.md) 참조.

## 모듈별 책임 분리 원칙

- ① 파서: 코퍼스별 — **데이터 추출만**, 마커는 text에 그대로 보존, 타임스탬프 토큰만 분리
- ② Cleaner: 코퍼스별 — **마커 처리 전담**, parsing·monologue 로직 없음
- ③ Monologue: 공통 — **턴 구조 처리 전담**, 어휘적 변경 없음
- ④ Pairs: 공통 — **외부 LLM 호출 전담**
- ⑤ Stats: 공통 — **read-only**, 데이터 변경 없음

각 모듈은 JSONL 입출력으로 결합되므로 단위 실행·재실행이 격리된다.

## 마커 처리 정책 (모듈 ② Cleaner)

마커 체계는 코퍼스별로 다르므로 cleaner도 어댑터별이다 (cf. [ADR-0006](../decisions/0006-adapter-structure-and-common-ir.md), [ADR-0011](../decisions/0011-corpus-adapter-interface.md)).

### CHAT (SBCSAE)

| 마커 | 의미 | 처리 |
|---|---|---|
| `(.)`, `(..)` | 짧은/긴 포즈 | `<pause:short>` / `<pause:long>` |
| `um`, `uh`, `you know`, `I mean`, `well` | 단어형 필러 | 보존 |
| `+/.` | 발화 중단 | 자연 종결 처리 |
| `+...` | 말끝 흐림 | `...` |
| `ʔuh` | 성문 폐쇄음 표기 | `uh` |
| `I:`, `u:m`, `perc:e:nt` | 모음 늘림 (vowel lengthening) | 알파벳 직후 `:` 제거 |
| `&=tsk`, `&=laugh`, `&=in`, `&=ex` | 비언어 어노테이션 | 제거 |
| `&{l=X ... &}l=X`, `&{n=X ... &}n=X`, `&{n=X}` | 스코프 마커 (코드스위칭/L2, 비언어음) | 외곽만 제거, 내부 발화 보존 (cf. ADR-0012) |
| `[% laugh]` | 코멘트 의존 티어 | 제거 (cf. ADR-0012) |
| `X`, `XX`, `XXX…` | unintelligible 음성 (단독 대문자 X 런) | 제거. `X-rays`/`XSes`/`X-`는 보존 (cf. ADR-0012) |
| `uh_you`, `U_S` | 밑줄 결합 발화 | `_`→공백 (cf. ADR-0012) |
| `⌈ ⌉`, `⌊n ⌋n` | 오버랩 마커 | 제거 |
| `760_1735` | 타임스탬프 | 파서가 분리 (cleaner 도달 전 제거됨) |

### Switchboard (MSU/ISIP)

PoC(`.work/switchboard-poc`) 실측 기반. 정규화 후 빈 발화(예: `[silence]` 전용 라인)는 ③ 도달 전 **drop**한다 (cf. [ADR-0009](../decisions/0009-switchboard-turn-reconstruction.md)).

| 마커 | 의미 | 처리 |
|---|---|---|
| `[silence]`, `[noise]`, `[laughter]`, `[vocalized-noise]` | 비언어/이벤트 | 제거 |
| `[laughter-yeah]` | 웃으며 발화한 단어 | → 단어만 (`yeah`) |
| `h[ow]-`, `tr[aveled]-` | 단어 재시작/중단 stub | 제거 (앞선 disfluency 단어는 보존) |
| `[bidness/business]` | 오발음 `[said/intended]` | → intended (슬래시 뒤) |
| `{alrighty}` | 신조어/비표준어 | 중괄호만 제거, 단어 보존 |
| `<b_aside>`, `<e_aside>` | 대화 외 발화 경계 | 마커 제거, 내부 텍스트 보존 |
| `because_1`, `them_1` | disambiguation 인덱스 `_<digit>` | `_\d+` 제거 → 기본형 |
| `AT&T`, `A&M` | 고유명사 | 보존 |

타임스탬프(초)는 파서에서 `t_start_ms`/`t_end_ms`로 분리되며, A/B 면은 `t_start_ms` 기준으로 인터리브된다 (cf. ADR-0009).

## 출력 디렉토리 구조

```
data/                                  # gitignore
├── parsed/<SOURCE>/<stem>.jsonl       # ① 파서 출력 (Utterance)
├── cleaned/<SOURCE>/<stem>.jsonl      # ② 정규화 출력 (Utterance)
├── monologues/<SOURCE>/<stem>.jsonl   # ③ 재조립 출력 (Monologue)
├── pairs/<SOURCE>/<stem>.jsonl        # ④ 역변환 출력 (Pair)
├── stats/<SOURCE>/<stem>.json         # ⑤ 통계
└── cache/pairs/<sha256>.json          # ④ LLM 응답 디스크 캐시 (cf. ADR-0007)
```

`<SOURCE>`는 corpus의 `SOURCE_NAME` (예: `SBCSAE`), `<stem>`은 파일 식별자(예: `SBC016`).

원본 CHA는 `datasets/`에 위치하며 다운로드 스크립트로 확보한다 (cf. [ADR-0002](../decisions/0002-sbcsae-license-policy.md)).

## 출력 페어 스키마 (모듈 ④ 산출물)

`Pair` dataclass (`scripttuner/preprocessing/ir.py`) 를 JSONL로 직렬화.

```python
@dataclass(frozen=True)
class Pair:
    pair_id: str          # e.g. "SBC016#mono_0001#casual#v2-zero-shot"
    source: str           # "SBCSAE"
    style: str            # "casual" (cf. ADR-0005)
    speaker: str
    spoken_text: str      # 원본 monologue.text (pause 토큰 포함, cf. ADR-0008)
    formal_text: str      # LLM 출력 (typography ASCII 정규화 적용)
    monologue_id: str     # 트레이스용
    metadata: dict        # model, prompt_version, prompt/completion_tokens, from_cache 등
```

JSON 예시:

```json
{
  "pair_id": "SBC016#mono_0002#casual#v2-zero-shot",
  "source": "SBCSAE",
  "style": "casual",
  "speaker": "TAMM",
  "spoken_text": "Um  <pause:long> because on a  <pause:short> ... you're not gonna be able ...",
  "formal_text": "Because on a tape deck like that you realize that you are not going to be able ...",
  "monologue_id": "SBC016#mono_0002",
  "metadata": {
    "model": "openai/gpt-oss-120b:free",
    "prompt_version": "v2-zero-shot",
    "prompt_tokens": 255,
    "completion_tokens": 67,
    "from_cache": false
  }
}
```

단순 페어 단위 메트릭(spoken/formal 토큰 수, 필러 빈도, pause 빈도 등)은 모듈 ⑤ Stats에서 별도 산출한다.

## 통계 출력 스키마 (모듈 ⑤ 산출물)

파일당 단일 JSON (`data/stats/<SOURCE>/<stem>.json`). 분포는 `{min, max, mean, median}` 객체.

```json
{
  "source": "SBCSAE",
  "n_pairs": 46,
  "n_unique_speakers": 2,
  "speakers": ["BRAD", "TAMM"],
  "spoken": {
    "tokens": {...},
    "fillers_per_pair": {...},
    "pause_short_per_pair": {...},
    "pause_long_per_pair": {...},
    "lexical_density": {...},
    "phrasal_verb_ratio": {...}
  },
  "formal": {
    "tokens": {...},
    "fillers_per_pair": {...},
    "lexical_density": {...},
    "phrasal_verb_ratio": {...}
  },
  "reduction_ratio": {...}
}
```

- **filler**: 단일어(`um, uh, well, like`) + 이중어(`you know, i mean, kind of, sort of`)
- **pause**: spoken_text의 `<pause:short>` / `<pause:long>` 카운트 (formal엔 부재)
- **lexical_density**: 내용어(NOUN/PROPN/VERB/ADJ/ADV) / 알파벳 토큰
- **phrasal_verb_ratio**: phrasal verb (verb + `prt` dep) / 전체 verb 수
- **reduction_ratio**: formal_tokens / spoken_tokens (페어별)

POS 기반 지표는 spacy(`en_core_web_sm`)에 의존. CLI에 `--no-pos`로 비활성화 가능. spacy 모델 설치는 `uv run python -m spacy download en_core_web_sm`.

## 향후 확장

- **Semi-formal 스타일 데이터 통합**: 동일 파이프라인 재사용, `style` 메타속성만 변경 (cf. [ADR-0005](../decisions/0005-style-as-dataset-metadata.md))
- **다른 코퍼스(Switchboard, BNC, TED 등) 통합**: `scripttuner/preprocessing/<format>/` 하위에 새 어댑터(parser + cleaner) 추가. ③~⑤ 재사용. IR 확장이 필요한 경우 `ir.py`의 `metadata` dict로 흡수 가능.
  - **Switchboard(OpenSLR #5)**: 턴 재구성 방식 결정됨 (cf. [ADR-0009](../decisions/0009-switchboard-turn-reconstruction.md), 라이선스 [ADR-0010](../decisions/0010-switchboard-license-policy.md)). CHAT과 달리 한 대화의 두 화자가 별도 파일(A/B)에 있고 상대 턴은 `[silence]`로 표시되므로, **어댑터(파서)가 양면을 `t_start_ms`로 인터리브해 단일 화자-교대 스트림을 복원**한다 → ③ `monologue.py`·[ADR-0004](../decisions/0004-backchannel-handling.md) 무변경 재사용. `[silence]`·비언어는 전부 drop(`<pause>` 합성 안 함), 빈 utterance는 ③ 도달 전 drop, 백채널 사전은 `build_monologues`에 주입. 파싱 단위 = 파일이 아니라 **대화(A+B → `swNNNN`)**.