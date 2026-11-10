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

## 2026-11-04 no_indoor_wd01 (Track A)
- changed: Exclude only assigned label 1 (indoors); restore labels 2, 4, 5, 6 and 7. Keep horizontal flips and weight_decay=0.1, all other recipe settings, seed 0 and the official training budget unchanged. Respect existing dropped.csv entries.
- why (student): "나름 잘했네 그럼 이번엔 실내라벨 데이터만 빼고 훈련돌려보자 트랙a로 colab" (after viewing a random sample of 100 label-1 photos).
- validation: 0.4831 (labels0389_no_flip: 0.4649; baseline seed 0: 0.4931; baseline seed 1: 0.4946). Official seed 0 run, 249,984 photo presentations, 39,756 selected photos, commit 74af675. Training took 2227.6 seconds (37.1 minutes) on CUDA. Imported from Downloads/from_colab (6).zip; result.json and experiments.csv agree. Recipe and selection hashes match local contents after newline normalization.
- what it means: Awaiting the result and the student's interpretation.
- next (student): "일단 이번 버전으로 릴리즈 한번 해줘봐 리더보드 점수 어떻게 나오나 함 보게". Release title chosen by the student: "실내 라벨만 제외, 좌우반전 유지, weight decay 0.1". Compared with labels0389_no_flip, selection, flips and weight decay all differ; compared with baseline, selection and weight decay differ. Neither comparison isolates indoor-photo exclusion. No completed labels0389_flip_wd01 run is recorded in experiments.csv.

## 보고서 작성 시 사후 보완: 학생이 제공한 동기와 해석

아래는 실험 당시에 기록된 것으로 가장하지 않고, 롱리포트 작성 중 학생이 설명한 내용을 정리한 것이다. 위의 원래 기록은 보존한다. 위 `Planned`와 미완료 표기는 오래된 상태이며, 현재 CSV에는 해당 완료 실행이 있다.

### labels0389_no_flip (Track A)
- changed: 1/2/4/5/6/7번 유형 제외와 좌우반전 제거, 26,501장.
- why (student): 교수님께 실내 예시와 비슷한 사진을 제거하는 방법을 안내받고, 도시를 구별할 특징이 없는 사진도 같은 방식으로 제거하면 도움이 될 것으로 예상했다. 도시당 300장씩 무작위 1,500장을 직접 라벨링했다.
- validation: 0.4649 (baseline seed 0/1: 0.4931/0.4946).
- what it means (student): 예상과 달리 약 3%p 낮아져 반전 제거가 문제였을 가능성을 생각했다. 한 번에 너무 많은 유형과 조건을 바꿔 원인을 알기 어려웠다. 가장 좋다고 생각한 조합부터 시도한 후 하나씩 확인할 계획이었지만 시간을 확보하지 못해 아쉬웠다.

### labels0389_flip_wd01 (Track A, 완료 상태 보완)
- changed: 반전 복원, WD 0.05에서 0.1로 증가. 선택은 26,501장 유지.
- why (student): 반전 제거가 낮은 점수의 원인인지 의심했고, 반복 노출에 따른 과적합을 억제하고 싶었다.
- validation: 0.4636 (previous: 0.4649).
- what it means (student): 반전을 복원해도 비슷한 점수였다. 두 조건이 함께 바뀌었으므로 반전이나 WD만의 효과를 확정하지 않는다.

### no_indoor_wd01 (Track A, 해석 보완)
- changed: 제외 범위를 실내 유형 1만으로 줄임. 39,756장, 반전 유지, WD 0.1.
- why (student): 여러 유형을 동시에 제외해 원인을 알 수 없어 우선 실내만 제외하는 방향으로 돌아갔다.
- validation: 0.4831 (baseline seed 0/1보다 1.00/1.15%p 낮음).
- what it means (student): 라벨링 접근에 문제가 있다고 받아들였다. 사람이 구별하기 어려운 사진 중에도 모델이 배울 특징이 있을 수 있고, 이를 함께 제거했을 가능성을 추측했다. 이는 확인된 원인이 아니다.

### no_indoor_wd01_balanced (Track A)
- changed: 실내 제외 39,756장, 반전, WD 0.1을 유지하고 도시 균형 샘플링 추가.
- why (student): 라벨링을 더 바꾸기보다 균형 샘플링에 초점을 옮겼다. 실제 실내 제외 선택은 계속 유지되었다.
- validation: 0.4999 (previous: 0.4831; baseline seed 0/1보다 0.68/0.53%p 높음).
- what it means (student): 데이터 품질을 가장 중요하게 생각했지만 균형 샘플링 같은 요소에도 집중할 필요를 느꼈다. 단일 seed 결과를 일반적 우월성으로 확정하지 않는다.

### b_baseline / b_mobilenet224 (Track B)
- changed: 모든 starter 학습 사진 사용. ResNet18 224에서 MobileNetV3-Large 224로 변경. 사전학습 가중치도 V1에서 V2로 달라짐.
- why (student): Track A에서 데이터 제외가 점수를 낮춰 전체 데이터를 사용했다. 다른 모델에서도 성능 차이가 나는지 궁금했다.
- validation: 0.4856 → 0.5145.
- what it means (student): 다른 모델에서 더 나은 점수를 얻어 모델을 바꾸어 보는 것이 의미 있다고 느꼈다.

### b_mobilenet224_balanced (Track B)
- changed: 도시 사진 수의 역수 가중치로 균형 샘플링 적용.
- why (student): 사진이 많은 도시의 특징을 더 많이 학습할 수 있다고 생각해 모든 도시가 비슷한 확률로 선택되도록 했다.
- validation: 0.5175 (previous: 0.5145).
- what it means (student): 평균은 조금 올랐지만 부산/오사카/타이베이 상승과 서울/도쿄 하락이 함께 있었다. 평균 점수 향상과 모든 도시를 고르게 분류하는 것은 다른 문제라고 느꼈다.

### b_mobilenet320_balanced / b_mobilenet512_balanced (Track B)
- changed: 균형 샘플링을 유지하고 크기를 224 → 320 → 512로 증가.
- why (student): 건물 형태, 창문, 간판 등 세부 특징이 도시 구분에 중요할 것으로 생각했다.
- validation: 0.5175 → 0.5496 → 0.5502. 훈련 시간 1605.3 → 2243.9 → 4274.8초.
- what it means (student): 320의 상승은 세부 정보가 도움이 되었을 수 있다는 예상과 부합했다. 512는 점수가 거의 같고 시간이 크게 늘어 해상도가 높을수록 반드시 좋은 것은 아니라는 점을 배웠다.

### 다음 계획 (학생)
일주일이 더 있다면 다양한 모델을 시도하고, 라벨링을 더 신중하게 다시 해보고 싶다. 이번에는 급하게 친구들에게 약 40장(확인된 파일 37장)을 받았으므로 더 많은 사람에게 사진을 받아 추가 훈련을 하고 싶다. 친구 사진 실험 결과는 이번 보고서 분석에서 제외한다.


## Track B release: track-b-2
- Student requested: MobileNetV3-Large 512 + 균형 샘플링.
- Artifact: downloaded original results/b_mobilenet512_balanced/model.onnx and submission.json from Google Drive; input dimensions checked as 512 x 512.
- Training validation: 0.5502, starter photos only, 4 epochs, balanced sampling.
- Local check: format and limits: OK; 4.20 million parameters, 1.11 GMACs. Full validation was attempted but interrupted by terminal-session termination; --no-val check completed with exit code 0. The 0.5502 score is from the original training record, not a completed local re-score.
```text
C:\Users\신혜원\Documents\mas1004-assignment2\submission\model.onnx: 16.8 MB, 4.20 million parameters, 1.11 billion multiply-adds per 512 x 512 photo
  limits: 12 million parameters, 2.0 billion multiply-adds
  format and limits: OK

```
- Model SHA256: 342841a489a4f2a9da83a09e2d0055ea257039c7795ed2e74d6e27b20b3e4f0a
