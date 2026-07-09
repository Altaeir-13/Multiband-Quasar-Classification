# Image Benchmark Results

Phase 1 controlled small benchmark using SDSS JPEG RGB cutouts. This report compares image classifiers across seeds 42, 123 and 13; it is still not the final large-scale experiment.

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

### resnet18 - `small-seed42-resnet18-pretrained`

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

### resnet18 - `small-seed123-resnet18-pretrained`

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

### resnet18 - `small-seed13-resnet18-pretrained`

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
