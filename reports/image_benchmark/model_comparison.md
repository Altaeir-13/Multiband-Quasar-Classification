# Image Benchmark Results

Phase 1 controlled image benchmark using SDSS JPEG RGB cutouts. Small runs compare seeds 42, 123 and 13; medium runs add controlled n_per_class=1000 repetitions. This is still not the final large-scale experiment.

The primary small benchmark uses from-scratch training for `simple_cnn`, `resnet18` and `densenet121`; a separate ResNet18 transfer-learning pilot uses `pretrained=true`. JPEG RGB cutouts are useful for this initial PDI benchmark, but they may not carry enough calibrated information to separate point-like QSO and STAR objects perfectly.

Smoke-test runs are omitted from the Markdown comparison; they remain in metrics.csv for traceability.

## Benchmark small - seed 42

| model       | run_type     | accuracy | precision_macro | recall_macro | f1_macro | best_epoch   | run_dir                                                        |
| ----------- | ------------ | -------- | --------------- | ------------ | -------- | ------------ | -------------------------------------------------------------- |
| simple_cnn  | from scratch | 0.6370   | 0.6675          | 0.6370       | 0.6279   | not_recorded | runs\image_benchmark\simple_cnn\small-simple-cnn               |
| resnet18    | from scratch | 0.8074   | 0.8297          | 0.8074       | 0.8056   | 2            | runs\image_benchmark\resnet18\small-resnet18                   |
| resnet18    | pretrained   | 0.7778   | 0.7850          | 0.7778       | 0.7797   | 4            | runs\image_benchmark\resnet18\small-seed42-resnet18-pretrained |
| densenet121 | from scratch | 0.7852   | 0.8062          | 0.7852       | 0.7761   | 4            | runs\image_benchmark\densenet121\small-densenet121             |

### simple_cnn - `small-simple-cnn`

| true\pred | STAR | GALAXY | QSO |
| --------- | ---- | ------ | --- |
| STAR      | 37   | 1      | 7   |
| GALAXY    | 19   | 19     | 7   |
| QSO       | 10   | 5      | 30  |

Dominant off-diagonal confusions: GALAXY -> STAR: 19; QSO -> STAR: 10; STAR -> QSO: 7; GALAXY -> QSO: 7.

### resnet18 - `small-resnet18`

| true\pred | STAR | GALAXY | QSO |
| --------- | ---- | ------ | --- |
| STAR      | 44   | 0      | 1   |
| GALAXY    | 5    | 35     | 5   |
| QSO       | 13   | 2      | 30  |

Dominant off-diagonal confusions: QSO -> STAR: 13; GALAXY -> STAR: 5; GALAXY -> QSO: 5; QSO -> GALAXY: 2.

### resnet18 pretrained - `small-seed42-resnet18-pretrained`

| true\pred | STAR | GALAXY | QSO |
| --------- | ---- | ------ | --- |
| STAR      | 36   | 2      | 7   |
| GALAXY    | 2    | 34     | 9   |
| QSO       | 3    | 7      | 35  |

Dominant off-diagonal confusions: GALAXY -> QSO: 9; STAR -> QSO: 7; QSO -> GALAXY: 7; QSO -> STAR: 3.

### densenet121 - `small-densenet121`

| true\pred | STAR | GALAXY | QSO |
| --------- | ---- | ------ | --- |
| STAR      | 44   | 1      | 0   |
| GALAXY    | 5    | 37     | 3   |
| QSO       | 12   | 8      | 25  |

Dominant off-diagonal confusions: QSO -> STAR: 12; QSO -> GALAXY: 8; GALAXY -> STAR: 5; GALAXY -> QSO: 3.

## Benchmark small - seed 123

| model       | run_type     | accuracy | precision_macro | recall_macro | f1_macro | best_epoch | run_dir                                                         |
| ----------- | ------------ | -------- | --------------- | ------------ | -------- | ---------- | --------------------------------------------------------------- |
| simple_cnn  | from scratch | 0.5481   | 0.5830          | 0.5481       | 0.5324   | 3          | runs\image_benchmark\simple_cnn\small-seed123-simple-cnn        |
| resnet18    | from scratch | 0.7407   | 0.7824          | 0.7407       | 0.7300   | 4          | runs\image_benchmark\resnet18\small-seed123-resnet18            |
| resnet18    | pretrained   | 0.7556   | 0.7680          | 0.7556       | 0.7528   | 7          | runs\image_benchmark\resnet18\small-seed123-resnet18-pretrained |
| densenet121 | from scratch | 0.7111   | 0.7328          | 0.7111       | 0.6936   | 6          | runs\image_benchmark\densenet121\small-seed123-densenet121      |

### simple_cnn - `small-seed123-simple-cnn`

| true\pred | STAR | GALAXY | QSO |
| --------- | ---- | ------ | --- |
| STAR      | 26   | 3      | 16  |
| GALAXY    | 14   | 14     | 17  |
| QSO       | 8    | 3      | 34  |

Dominant off-diagonal confusions: GALAXY -> QSO: 17; STAR -> QSO: 16; GALAXY -> STAR: 14; QSO -> STAR: 8.

### resnet18 - `small-seed123-resnet18`

| true\pred | STAR | GALAXY | QSO |
| --------- | ---- | ------ | --- |
| STAR      | 42   | 2      | 1   |
| GALAXY    | 9    | 36     | 0   |
| QSO       | 7    | 16     | 22  |

Dominant off-diagonal confusions: QSO -> GALAXY: 16; GALAXY -> STAR: 9; QSO -> STAR: 7; STAR -> GALAXY: 2.

### resnet18 pretrained - `small-seed123-resnet18-pretrained`

| true\pred | STAR | GALAXY | QSO |
| --------- | ---- | ------ | --- |
| STAR      | 40   | 1      | 4   |
| GALAXY    | 9    | 34     | 2   |
| QSO       | 9    | 8      | 28  |

Dominant off-diagonal confusions: GALAXY -> STAR: 9; QSO -> STAR: 9; QSO -> GALAXY: 8; STAR -> QSO: 4.

### densenet121 - `small-seed123-densenet121`

| true\pred | STAR | GALAXY | QSO |
| --------- | ---- | ------ | --- |
| STAR      | 42   | 0      | 3   |
| GALAXY    | 11   | 19     | 15  |
| QSO       | 6    | 4      | 35  |

Dominant off-diagonal confusions: GALAXY -> QSO: 15; GALAXY -> STAR: 11; QSO -> STAR: 6; QSO -> GALAXY: 4.

## Benchmark small - seed 13

| model       | run_type     | accuracy | precision_macro | recall_macro | f1_macro | best_epoch | run_dir                                                        |
| ----------- | ------------ | -------- | --------------- | ------------ | -------- | ---------- | -------------------------------------------------------------- |
| simple_cnn  | from scratch | 0.6444   | 0.6622          | 0.6444       | 0.6402   | 5          | runs\image_benchmark\simple_cnn\small-seed13-simple-cnn        |
| resnet18    | from scratch | 0.6000   | 0.6809          | 0.6000       | 0.6035   | 1          | runs\image_benchmark\resnet18\small-seed13-resnet18            |
| resnet18    | pretrained   | 0.7704   | 0.7722          | 0.7704       | 0.7696   | 6          | runs\image_benchmark\resnet18\small-seed13-resnet18-pretrained |
| densenet121 | from scratch | 0.5704   | 0.6304          | 0.5704       | 0.4881   | 1          | runs\image_benchmark\densenet121\small-seed13-densenet121      |

### simple_cnn - `small-seed13-simple-cnn`

| true\pred | STAR | GALAXY | QSO |
| --------- | ---- | ------ | --- |
| STAR      | 37   | 2      | 6   |
| GALAXY    | 15   | 26     | 4   |
| QSO       | 12   | 9      | 24  |

Dominant off-diagonal confusions: GALAXY -> STAR: 15; QSO -> STAR: 12; QSO -> GALAXY: 9; STAR -> QSO: 6.

### resnet18 - `small-seed13-resnet18`

| true\pred | STAR | GALAXY | QSO |
| --------- | ---- | ------ | --- |
| STAR      | 24   | 1      | 20  |
| GALAXY    | 2    | 21     | 22  |
| QSO       | 2    | 7      | 36  |

Dominant off-diagonal confusions: GALAXY -> QSO: 22; STAR -> QSO: 20; QSO -> GALAXY: 7; GALAXY -> STAR: 2.

### resnet18 pretrained - `small-seed13-resnet18-pretrained`

| true\pred | STAR | GALAXY | QSO |
| --------- | ---- | ------ | --- |
| STAR      | 38   | 0      | 7   |
| GALAXY    | 6    | 35     | 4   |
| QSO       | 7    | 7      | 31  |

Dominant off-diagonal confusions: STAR -> QSO: 7; QSO -> STAR: 7; QSO -> GALAXY: 7; GALAXY -> STAR: 6.

### densenet121 - `small-seed13-densenet121`

| true\pred | STAR | GALAXY | QSO |
| --------- | ---- | ------ | --- |
| STAR      | 35   | 9      | 1   |
| GALAXY    | 6    | 39     | 0   |
| QSO       | 16   | 26     | 3   |

Dominant off-diagonal confusions: QSO -> GALAXY: 26; QSO -> STAR: 16; STAR -> GALAXY: 9; GALAXY -> STAR: 6.

## Benchmark medium - seed 42

| model       | run_type     | accuracy | precision_macro | recall_macro | f1_macro | best_epoch | run_dir                                                  |
| ----------- | ------------ | -------- | --------------- | ------------ | -------- | ---------- | -------------------------------------------------------- |
| simple_cnn  | from scratch | 0.7356   | 0.7670          | 0.7356       | 0.7217   | 5          | runs\image_benchmark\simple_cnn\medium-simple-cnn        |
| resnet18    | pretrained   | 0.8933   | 0.8980          | 0.8933       | 0.8938   | 4          | runs\image_benchmark\resnet18\medium-resnet18-pretrained |
| densenet121 | from scratch | 0.8778   | 0.8781          | 0.8778       | 0.8778   | 4          | runs\image_benchmark\densenet121\medium-densenet121      |

### simple_cnn - `medium-simple-cnn`

| true\pred | STAR | GALAXY | QSO |
| --------- | ---- | ------ | --- |
| STAR      | 137  | 7      | 6   |
| GALAXY    | 23   | 125    | 2   |
| QSO       | 26   | 55     | 69  |

Dominant off-diagonal confusions: QSO -> GALAXY: 55; QSO -> STAR: 26; GALAXY -> STAR: 23; STAR -> GALAXY: 7.

### resnet18 pretrained - `medium-resnet18-pretrained`

| true\pred | STAR | GALAXY | QSO |
| --------- | ---- | ------ | --- |
| STAR      | 136  | 0      | 14  |
| GALAXY    | 8    | 127    | 15  |
| QSO       | 4    | 7      | 139 |

Dominant off-diagonal confusions: GALAXY -> QSO: 15; STAR -> QSO: 14; GALAXY -> STAR: 8; QSO -> GALAXY: 7.

### densenet121 - `medium-densenet121`

| true\pred | STAR | GALAXY | QSO |
| --------- | ---- | ------ | --- |
| STAR      | 130  | 7      | 13  |
| GALAXY    | 11   | 131    | 8   |
| QSO       | 4    | 12     | 134 |

Dominant off-diagonal confusions: STAR -> QSO: 13; QSO -> GALAXY: 12; GALAXY -> STAR: 11; GALAXY -> QSO: 8.

## Benchmark medium - seed 123

| model       | run_type     | accuracy | precision_macro | recall_macro | f1_macro | best_epoch | run_dir                                                          |
| ----------- | ------------ | -------- | --------------- | ------------ | -------- | ---------- | ---------------------------------------------------------------- |
| resnet18    | pretrained   | 0.9000   | 0.9010          | 0.9000       | 0.8998   | 7          | runs\image_benchmark\resnet18\medium-seed123-resnet18-pretrained |
| densenet121 | from scratch | 0.5867   | 0.7247          | 0.5867       | 0.5189   | 5          | runs\image_benchmark\densenet121\medium-seed123-densenet121      |

### resnet18 pretrained - `medium-seed123-resnet18-pretrained`

| true\pred | STAR | GALAXY | QSO |
| --------- | ---- | ------ | --- |
| STAR      | 138  | 0      | 12  |
| GALAXY    | 5    | 142    | 3   |
| QSO       | 18   | 7      | 125 |

Dominant off-diagonal confusions: QSO -> STAR: 18; STAR -> QSO: 12; QSO -> GALAXY: 7; GALAXY -> STAR: 5.

### densenet121 - `medium-seed123-densenet121`

| true\pred | STAR | GALAXY | QSO |
| --------- | ---- | ------ | --- |
| STAR      | 149  | 0      | 1   |
| GALAXY    | 44   | 104    | 2   |
| QSO       | 131  | 8      | 11  |

Dominant off-diagonal confusions: QSO -> STAR: 131; GALAXY -> STAR: 44; QSO -> GALAXY: 8; GALAXY -> QSO: 2.

## Benchmark medium - seed 13

| model       | run_type     | accuracy | precision_macro | recall_macro | f1_macro | best_epoch | run_dir                                                         |
| ----------- | ------------ | -------- | --------------- | ------------ | -------- | ---------- | --------------------------------------------------------------- |
| resnet18    | pretrained   | 0.8600   | 0.8609          | 0.8600       | 0.8591   | 4          | runs\image_benchmark\resnet18\medium-seed13-resnet18-pretrained |
| densenet121 | from scratch | 0.6933   | 0.7452          | 0.6933       | 0.6819   | 5          | runs\image_benchmark\densenet121\medium-seed13-densenet121      |

### resnet18 pretrained - `medium-seed13-resnet18-pretrained`

| true\pred | STAR | GALAXY | QSO |
| --------- | ---- | ------ | --- |
| STAR      | 136  | 0      | 14  |
| GALAXY    | 7    | 136    | 7   |
| QSO       | 22   | 13     | 115 |

Dominant off-diagonal confusions: QSO -> STAR: 22; STAR -> QSO: 14; QSO -> GALAXY: 13; GALAXY -> STAR: 7.

### densenet121 - `medium-seed13-densenet121`

| true\pred | STAR | GALAXY | QSO |
| --------- | ---- | ------ | --- |
| STAR      | 135  | 0      | 15  |
| GALAXY    | 24   | 66     | 60  |
| QSO       | 35   | 4      | 111 |

Dominant off-diagonal confusions: GALAXY -> QSO: 60; QSO -> STAR: 35; GALAXY -> STAR: 24; STAR -> QSO: 15.

### Small vs medium - seed 42

| run                 | small F1 | medium F1 | delta F1 | small acc | medium acc | delta acc |
| ------------------- | -------- | --------- | -------- | --------- | ---------- | --------- |
| simple_cnn          | 0.6279   | 0.7217    | 0.0939   | 0.6370    | 0.7356     | 0.0985    |
| resnet18 pretrained | 0.7797   | 0.8938    | 0.1141   | 0.7778    | 0.8933     | 0.1156    |
| densenet121         | 0.7761   | 0.8778    | 0.1017   | 0.7852    | 0.8778     | 0.0926    |

Best medium seed 42 model by test F1 macro: `resnet18 pretrained` with F1=`0.8938` and accuracy=`0.8933`.

`resnet18 pretrained` remains the primary baseline candidate at medium scale with F1=`0.8938`.

Medium seed 42 improved F1 over small seed 42 for: `simple_cnn`, `resnet18 pretrained`, `densenet121`.

Medium now has repeated seeds for `resnet18 pretrained`; use the robustness section for the variance conclusion.

Inspect QSO -> STAR, QSO -> GALAXY and GALAXY -> STAR confusions before either scaling to n_per_class=2000 or adding new architectures.

## Medium robustness - ResNet18 pretrained

| run                 | F1 seed 42 | F1 seed 123 | F1 seed 13 | mean F1 | std F1 | acc seed 42 | acc seed 123 | acc seed 13 | mean acc | std acc |
| ------------------- | ---------- | ----------- | ---------- | ------- | ------ | ----------- | ------------ | ----------- | -------- | ------- |
| resnet18 pretrained | 0.8938     | 0.8998      | 0.8591     | 0.8842  | 0.0179 | 0.8933      | 0.9000       | 0.8600      | 0.8844   | 0.0175  |

### Test confusion matrix - seed 42 - `medium-resnet18-pretrained`

| true\pred | STAR | GALAXY | QSO |
| --------- | ---- | ------ | --- |
| STAR      | 136  | 0      | 14  |
| GALAXY    | 8    | 127    | 15  |
| QSO       | 4    | 7      | 139 |

### Test confusion matrix - seed 123 - `medium-seed123-resnet18-pretrained`

| true\pred | STAR | GALAXY | QSO |
| --------- | ---- | ------ | --- |
| STAR      | 138  | 0      | 12  |
| GALAXY    | 5    | 142    | 3   |
| QSO       | 18   | 7      | 125 |

### Test confusion matrix - seed 13 - `medium-seed13-resnet18-pretrained`

| true\pred | STAR | GALAXY | QSO |
| --------- | ---- | ------ | --- |
| STAR      | 136  | 0      | 14  |
| GALAXY    | 7    | 136    | 7   |
| QSO       | 22   | 13     | 115 |

Seed 42 was representative for this medium pretrained ResNet18 run: its F1 differs from the three-seed mean by +0.0095.

Seed 13 is the lower-tail sample in this repetition (delta from mean F1 -0.0251), so seed 42 alone was mildly optimistic.

`resnet18 pretrained` remains stable at medium scale: all three runs stay between F1=0.8591 and F1=0.8998, with std F1=0.0179.

Compared with the small pretrained ResNet18 pilot, medium did not reduce seed-to-seed F1 variance (small std 0.0111, medium std 0.0179). It did raise the mean performance, so scale improved level more clearly than variance.

Repeat `densenet121` on medium seeds 123 and 13 before final model selection, because seed 42 was close to ResNet18.

Do not prioritize another `simple_cnn` repetition now; it is useful as a floor baseline but is not the leading candidate.

Scaling to n_per_class=2000 is reasonable after the DenseNet robustness check, keeping the model set fixed.

## Medium robustness - DenseNet121

| run         | F1 seed 42 | F1 seed 123 | F1 seed 13 | mean F1 | std F1 | acc seed 42 | acc seed 123 | acc seed 13 | mean acc | std acc |
| ----------- | ---------- | ----------- | ---------- | ------- | ------ | ----------- | ------------ | ----------- | -------- | ------- |
| densenet121 | 0.8778     | 0.5189      | 0.6819     | 0.6929  | 0.1467 | 0.8778      | 0.5867       | 0.6933      | 0.7193   | 0.1203  |

### Train/validation summary - seed 42 - medium-densenet121

Best epoch: 4. At the best epoch, train loss was 0.3952, validation loss was 0.3872, validation accuracy was 0.8600 and validation F1 macro was 0.8587. Test accuracy was 0.8778 and test F1 macro was 0.8778.

| true\pred | STAR | GALAXY | QSO |
| --------- | ---- | ------ | --- |
| STAR      | 130  | 7      | 13  |
| GALAXY    | 11   | 131    | 8   |
| QSO       | 4    | 12     | 134 |

### Train/validation summary - seed 123 - medium-seed123-densenet121

Best epoch: 5. At the best epoch, train loss was 0.3752, validation loss was 1.1743, validation accuracy was 0.6267 and validation F1 macro was 0.5776. Test accuracy was 0.5867 and test F1 macro was 0.5189.

| true\pred | STAR | GALAXY | QSO |
| --------- | ---- | ------ | --- |
| STAR      | 149  | 0      | 1   |
| GALAXY    | 44   | 104    | 2   |
| QSO       | 131  | 8      | 11  |

### Train/validation summary - seed 13 - medium-seed13-densenet121

Best epoch: 5. At the best epoch, train loss was 0.3632, validation loss was 0.6920, validation accuracy was 0.7311 and validation F1 macro was 0.7148. Test accuracy was 0.6933 and test F1 macro was 0.6819.

| true\pred | STAR | GALAXY | QSO |
| --------- | ---- | ------ | --- |
| STAR      | 135  | 0      | 15  |
| GALAXY    | 24   | 66     | 60  |
| QSO       | 35   | 4      | 111 |

DenseNet121 is not competitive with 
esnet18 pretrained on medium robustness. Seed 42 looked close to ResNet18, but the repeated seeds expose high variance: DenseNet mean F1 is 0.6929 versus 0.8842 for pretrained ResNet18, and DenseNet F1 std is 0.1467 versus 0.0179.

The failure modes are seed-dependent. Seed 123 collapses QSO recall, sending 131/150 QSO examples to STAR and recovering only 11/150 QSO. Seed 13 recovers QSO better than seed 123, but GALAXY recall collapses, with 60/150 GALAXY predicted as QSO and 24/150 as STAR. By contrast, pretrained ResNet18 keeps all three medium seeds in the F1 range 0.8591-0.8998.

Recommendation: do not carry DenseNet121 as a primary model into the next scale. Keep 
esnet18 pretrained as the main Phase 1 image baseline; DenseNet121 can remain documented as a negative robustness result, but the next medium/large compute should not prioritize it.

## Medium model synthesis

| model               | seeds available | mean F1 | std F1 | mean acc | std acc |
| ------------------- | --------------- | ------- | ------ | -------- | ------- |
| resnet18 pretrained | 42/123/13       | 0.8842  | 0.0179 | 0.8844   | 0.0175  |
| densenet121         | 42/123/13       | 0.6929  | 0.1467 | 0.7193   | 0.1203  |
| simple_cnn          | 42 only         | 0.7217  | n/a    | 0.7356   | n/a     |

The best medium model by mean F1 is 
esnet18 pretrained. The most stable model among models with three medium seeds is also 
esnet18 pretrained, with F1 std 0.0179. simple_cnn has only seed 42 on medium, so it does not yet have a medium robustness estimate.

Across the targeted confusions, pretrained ResNet18 is consistently less pathological than DenseNet121. ResNet18 has QSO -> STAR counts of 4, 18 and 22; DenseNet has 4, 131 and 35. ResNet18 has QSO -> GALAXY counts of 7, 7 and 13; DenseNet has 12, 8 and 4. ResNet18 has GALAXY -> STAR counts of 8, 5 and 7; DenseNet has 11, 44 and 24. The decisive difference is not QSO -> GALAXY, where DenseNet is sometimes lower, but the severe DenseNet QSO -> STAR and GALAXY -> STAR/GALAXY -> QSO instability across seeds.

For the next Phase 1 step, prioritize scaling 
esnet18 pretrained to n_per_class=2000. Repeating simple_cnn on medium seeds 123/13 is lower value unless a complete baseline floor is needed. Do not add MobileNet, ResNet50 or EfficientNet yet; new architectures would expand the search before resolving the scale question for the current winner. After the n_per_class=2000 image-only run is complete, decide whether Phase 1 has enough evidence to close and then prepare Phase 2 separately.

## Benchmark From Scratch - Aggregate F1 Macro By Model

| model       | F1 seed 42 | F1 seed 123 | F1 seed 13 | mean F1 | std F1 |
| ----------- | ---------- | ----------- | ---------- | ------- | ------ |
| simple_cnn  | 0.6279     | 0.5324      | 0.6402     | 0.6002  | 0.0482 |
| resnet18    | 0.8056     | 0.7300      | 0.6035     | 0.7130  | 0.0834 |
| densenet121 | 0.7761     | 0.6936      | 0.4881     | 0.6526  | 0.1211 |

## Benchmark From Scratch - Aggregate Accuracy By Model

| model       | acc seed 42 | acc seed 123 | acc seed 13 | mean acc | std acc |
| ----------- | ----------- | ------------ | ----------- | -------- | ------- |
| simple_cnn  | 0.6370      | 0.5481       | 0.6444      | 0.6099   | 0.0438  |
| resnet18    | 0.8074      | 0.7407       | 0.6000      | 0.7160   | 0.0865  |
| densenet121 | 0.7852      | 0.7111       | 0.5704      | 0.6889   | 0.0891  |

## Transfer Learning Pilot - ResNet18

| run                   | F1 seed 42 | F1 seed 123 | F1 seed 13 | mean F1 | std F1 | acc seed 42 | acc seed 123 | acc seed 13 | mean acc | std acc |
| --------------------- | ---------- | ----------- | ---------- | ------- | ------ | ----------- | ------------ | ----------- | -------- | ------- |
| resnet18 from scratch | 0.8056     | 0.7300      | 0.6035     | 0.7130  | 0.0834 | 0.8074      | 0.7407       | 0.6000      | 0.7160   | 0.0865  |
| resnet18 pretrained   | 0.7797     | 0.7528      | 0.7696     | 0.7674  | 0.0111 | 0.7778      | 0.7556       | 0.7704      | 0.7679   | 0.0092  |

`resnet18 pretrained` achieved higher mean F1 than `resnet18` from scratch (0.7674 vs 0.7130, delta +0.0543).

`resnet18 pretrained` strongly reduced variation across seeds (F1 std 0.0111 vs 0.0834, reduction 0.0723).

The main gain came from recovering seed 13 (F1 0.7696 vs 0.6035, delta +0.1661).

This result suggests using pretrained image backbones as the default for the next scale of the image-classification benchmark.

It still does not support any conclusion about photometric redshift, because Phase 1 is only image classification.

## From-Scratch Benchmark Interpretation

Best model by mean test F1 macro: `resnet18` with mean F1=`0.7130`.

Most stable model by lowest observed test F1 standard deviation: `simple_cnn` with std F1=`0.0482`.

`simple_cnn` remains useful as a minimum baseline, but it trails the stronger architectures in mean test F1. `resnet18` should remain the principal baseline only if its mean performance and variance stay competitive as the dataset grows. `densenet121` is competitive and should be compared on equal seeds before selecting a final image baseline.

Recurring confusions to inspect before scaling are QSO -> STAR, QSO -> GALAXY and GALAXY -> STAR. These errors are scientifically plausible with JPEG RGB cutouts because QSO and STAR can both appear point-like, and color/photometric calibration is limited compared with FITS or tabular photometry.

No conclusion about photometric redshift should be drawn from this report. Phase 1 is only image classification; redshift estimation is intentionally outside the current benchmark scope.
