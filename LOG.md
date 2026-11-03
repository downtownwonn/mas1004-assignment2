# Log

One entry for each recipe, selection or Track B model you train, newest at
the bottom. The numbers come from `experiments.csv`. "Why" and "what it means"
are in your own words: your agent asks you for them and does not make them up.
End the log with your plan for what to try next.

<!-- The form of an entry:

## <date> <tag> (Track A or B)
- changed: what is different from the run before
- why: your reason
- validation: 0.000 (before: 0.000)
- what it means: what you think the result says
-->

## 2026-11-04 labels0389_no_flip (Track A)
- changed: Assigned labels 1, 2, 4, 5, 6 and 7 excluded; 26,501 photos retained. Removed horizontal flips; other recipe settings unchanged. Human labels take priority over automatic assignments.
- why: Original motivation for these two changes has not yet been provided by the student.
- validation: 0.4649 (baseline seed 0: 0.4931; baseline seed 1: 0.4946). Official seed 0 run, 249,984 photo presentations, commit 1b98a91.
- what it means (student): "내 생각에는 4번이 유력한 원인이거든 사진 수는 줄었는데 학습횟수는 그대로라 같은 사진을 여러번 봤다는거말이야"
- interpretation status: This is the student's hypothesis, not an established cause; filtering and flip removal were changed together.
- next (student): Release this version to observe the leaderboard, then test an improvement addressing repeated exposure. The next method is not yet selected; Track A fixes the training budget at 250,000 photos.

## Planned: labels0389_flip_wd01 (Track A)
- changed: Keep the same 26,501-photo selection; restore RandomHorizontalFlip (default probability 0.5); increase AdamW weight_decay from 0.05 to 0.1. Other recipe settings and the official budget unchanged; seed 0 only.
- why (student): "좌우반전 제거하지말고 weightdecay를 0.1로 높여서 과도 맞춤 억제하는 식으로 다시 학습가자"
- validation: Pending training (previous run: 0.4649).
- what it means: Awaiting the result and the student's interpretation. Both flip and weight decay differ from the previous run, so this comparison does not isolate weight decay.
- next: Run the quick check, then one full run on Colab; compare against labels0389_no_flip and ask the student what the result means.

## Planned: no_indoor_wd01 (Track A)
- changed: Exclude only assigned label 1 (indoors); restore labels 2, 4, 5, 6 and 7. Keep horizontal flips and weight_decay=0.1, all other recipe settings, seed 0 and the official training budget unchanged. Respect existing dropped.csv entries.
- why (student): "나름 잘했네 그럼 이번엔 실내라벨 데이터만 빼고 훈련돌려보자 트랙a로 colab" (after viewing a random sample of 100 label-1 photos).
- validation: Pending training; no result assumed for labels0389_flip_wd01.
- what it means: Awaiting the result and the student's interpretation.
- next: Quick check followed by one full Colab run. Queue replaces the previous commands; it does not rerun the previous selection.
