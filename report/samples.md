# 정성 샘플 — 재구두점 before/after

직전 baseline `t5gemma2-1b-SBCSAE-lora-es`(Tier-A target)와 최종
`t5gemma2-1b-SBCSAE-lora-repunct`(재구두점 target)의 **동일 입력**에 대한 test split 예측.
두 모델은 학습 길이·하이퍼파라미터가 같고 **target 데이터만 다르다** → 차이는 재구두점 효과.

- **입력(formal)**: 파이프라인 `pairs` 단계에서 LLM이 생성한 문어체 문장
- **BEFORE**: `-es` 예측 / **AFTER**: repunct 예측
- 표기: `<pause:short>` / `<pause:long>`은 발화 휴지(쉼) 마커
- ⚠️ 참조(reference, SBCSAE 원문 발화)는 라이선스(CC BY-ND)상 본 문서에서 생략.
  분포 비교는 [`metrics_summary.md`](metrics_summary.md) 참조.

각 예시에서 BEFORE의 ① 마침표 앞 공백(`' .'`), ② 쉼표 부재를 확인하고,
AFTER에서 둘 다 교정되며 절 경계에 쉼표가 자연스럽게 들어가는지 본다.

---

**1.**
> **입력:** Cause, instead of just sitting in the class and getting five dollars an hour, I was now going to be up there teaching it. And instead of getting five dollars an hour, I ended up getting fifteen.
>
> **BEFORE:** cause instead of just like sitting in the class and get like five dollars an hour I was now gonna be up there teaching it `.` And instead of getting five dollars an hour I ended up getting fifteen `.`
>
> **AFTER:** Cause, `<pause:short>` um, instead of `<pause:short>` just sitting in the class and getting five dollars an hour, `<pause:short>` I was now `<pause:short>` gonna be up there teaching it, and instead of getting five dollars an hour, I ended up getting fifteen.

**2.**
> **입력:** And it's lovely. The people are sweet and nice; they're little, they're dark, they all wear white, and the city is quite nice. It's a very manageable size, Merida.
>
> **BEFORE:** `<pause:long>` And it's lovely `.` `<pause:long>` the people are sweet and nice `<pause:short>` they're little they're dark they all wear white `<pause:short>` and the city is quite nice `.` It's a very manageable size Merida `.`
>
> **AFTER:** And it's lovely. `<pause:short>` The people are sweet and nice, they're little, they're dark, `<pause:long>` they all wear white, and the city is quite nice. `<pause:short>` It's a very manageable size, Merida.

**3.**
> **입력:** I remember hearing some guys complaining years ago, when we first moved here, that they call it Twin Cities, but they let all these people come in and take over all the first chairs and all this stuff.
>
> **BEFORE:** I remember hearing some guys complaining years ago when we first moved here that they call it Twin Cities but they let all these people come in and take over all the first chairs and all this stuff `.`
>
> **AFTER:** I remember hearing some guys complain, years ago `<pause:long>` when we first moved here, that they call it Twin Cities, but they let all these people come in and take over all the first chairs and all this stuff.

**4.** (물음표 복원 예)
> **입력:** And I'll tell them, "Alright, sit down." You know, I had to get really tough in the last week. The first week I played with them all week long, which was really stupid because they got worked up.
>
> **BEFORE:** and I'll tell em alright sit down `.` `<pause:long>` You know I had to get really tough in the last week `.` The first week I played with em all week long which was really stupid cause they got worked up `.`
>
> **AFTER:** And I'll tell them, okay, sit down, you know, and I had to get really tough in the last week, I had, `<pause:long>` like the first week I played with them all week long? Which was `<pause:short>` was real stupid because they got worked up.

---

관찰: BEFORE는 마침표 앞 공백(`' .'`)이 남고 쉼표가 전혀 없어 절을 마침표로 끊거나
이어붙인다. AFTER는 공백 아티팩트가 사라지고 절 경계에 쉼표가 들어가 사람 발화의
호흡에 가까워진다(4번은 `?`까지 복원). 동시에 `gonna`·`em`/`them` 축약, filler·pause
같은 구어 특징은 그대로 유지된다 — 구두점만 고치고 구어성은 보존.

---

# 정성 샘플 — 스타일 제어 토큰 분기 (통합 모델)

통합 모델 `t5gemma2-1b-COMBINED-lora`의 **동일 입력에 style 토큰만 바꿔** 생성한
casual vs semi_formal 출력. 같은 입력이므로 차이는 순수하게 제어 토큰 효과다.

- **CASUAL**: `<style:casual>` 출력 / **SEMI**: `<style:semi_formal>` 출력
- 두 스타일의 핵심 차이는 ① **문장 분절**(casual은 짧게 토막, semi는 길게 이어붙임),
  ② **pause**(casual만 방출). 분포 수치는 [`metrics_summary.md`](metrics_summary.md) 참조.

---

**1.** (분절 + pause 대비)
> **입력:** Alright. Because that would be really handy then, we can actually test it on the user, as opposed to just doing it and not using it. Because if you just have the segmentation, that's great, but we cannot compare it to the hand annotated, you know, the hand segmented tool.
>
> **CASUAL:** `<pause:long>` Alright. `<pause:short>` Because that would be really handy then, `<pause:long>` we can actually test it on the user. `<pause:short>` As opposed to `<pause:long>` just doing it and not using it. `<pause:long>` Because if you just have the segmentation. `<pause:short>` That's great. `<pause:short>` But we can't `<pause:long>` compare that to the hand annotated. `<pause:short>` You know, the hand segmented tool.
>
> **SEMI:** Yeah. Alright. Because that would be really handy then, then we can actually test it on the user as opposed to just doing it and not using it. Because if you just got the segmentation, that's great, but we can't compare to the hand annotated, you know, the hand segmented tool.

**2.** (이어붙이기 대비)
> **입력:** She said she thought she had gotten a brand inspection done since she got to Colorado, but then she said she didn't know where it was, and that's what he wanted to know.
>
> **CASUAL:** `<pause:long>` She said she thought she'd gotten a brand inspection done `<pause:short>` since she got to Colorado but then she said `<pause:short>` she didn't know where it was and that's what he wanted to know and. `<pause:short>` Uh.
>
> **SEMI:** She said she thought she had gotten a brand inspection done since she got to Colorado, but then she said she didn't know where it was and that's what he wanted to know.

---

관찰: 같은 내용을 CASUAL은 pause를 박으며 짧은 문장으로 토막내고(1번: 8문장),
SEMI는 pause 없이 절을 쉼표로 이어 한 흐름으로 묶는다(1번: 2~3문장). 1번 CASUAL의
`That's great.` / `But we can't compare...` 처럼 절을 마침표로 끊는 vs SEMI의
`that's great, but we can't compare...` 쉼표 연결이 분절 차이를 직접 보여준다.
제어 토큰이 동일 입력에서 두 발화 스타일을 분기시킨다.