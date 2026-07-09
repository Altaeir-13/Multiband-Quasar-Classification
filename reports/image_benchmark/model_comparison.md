# Image Benchmark Results

Partial Phase 1 image benchmark on the controlled small SDSS sample. Only models with available metrics are listed; this is not the final large-scale experiment.

All currently configured small runs use SDSS JPEG RGB cutouts and `pretrained=false` unless a metric row explicitly records otherwise. JPEG RGB may not contain enough information to perfectly separate STAR, GALAXY and QSO, especially for point-like objects.

Smoke-test runs are omitted from the main comparison table; they remain in metrics.csv for traceability.

## Aggregate Metrics

| model       | split | accuracy | precision_macro | recall_macro | f1_macro | pretrained   | best_epoch   | run_dir                                            |
| ----------- | ----- | -------- | --------------- | ------------ | -------- | ------------ | ------------ | -------------------------------------------------- |
| resnet18    | test  | 0.8074   | 0.8297          | 0.8074       | 0.8056   | False        | 2            | runs\image_benchmark\resnet18\small-resnet18       |
| densenet121 | test  | 0.7852   | 0.8062          | 0.7852       | 0.7761   | False        | 4            | runs\image_benchmark\densenet121\small-densenet121 |
| simple_cnn  | test  | 0.6370   | 0.6675          | 0.6370       | 0.6279   | not_recorded | not_recorded | runs\image_benchmark\simple_cnn\small-simple-cnn   |
| resnet18    | val   | 0.8000   | 0.8068          | 0.8000       | 0.7988   | False        | 2            | runs\image_benchmark\resnet18\small-resnet18       |
| densenet121 | val   | 0.7778   | 0.8015          | 0.7778       | 0.7711   | False        | 4            | runs\image_benchmark\densenet121\small-densenet121 |
| simple_cnn  | val   | 0.6444   | 0.6777          | 0.6444       | 0.6350   | not_recorded | not_recorded | runs\image_benchmark\simple_cnn\small-simple-cnn   |

## Best Model

Best test F1 macro: `resnet18` with F1=`0.8056` and accuracy=`0.8074`.

## Test Confusion Matrices

### densenet121

| true\pred | STAR | GALAXY | QSO |
| --------- | ---- | ------ | --- |
| STAR      | 44   | 1      | 0   |
| GALAXY    | 5    | 37     | 3   |
| QSO       | 12   | 8      | 25  |

Dominant off-diagonal confusions: QSO -> STAR: 12; QSO -> GALAXY: 8; GALAXY -> STAR: 5; GALAXY -> QSO: 3.

### resnet18

| true\pred | STAR | GALAXY | QSO |
| --------- | ---- | ------ | --- |
| STAR      | 44   | 0      | 1   |
| GALAXY    | 5    | 35     | 5   |
| QSO       | 13   | 2      | 30  |

Dominant off-diagonal confusions: QSO -> STAR: 13; GALAXY -> STAR: 5; GALAXY -> QSO: 5; QSO -> GALAXY: 2.

### simple_cnn

| true\pred | STAR | GALAXY | QSO |
| --------- | ---- | ------ | --- |
| STAR      | 37   | 1      | 7   |
| GALAXY    | 19   | 19     | 7   |
| QSO       | 10   | 5      | 30  |

Dominant off-diagonal confusions: GALAXY -> STAR: 19; QSO -> STAR: 10; STAR -> QSO: 7; GALAXY -> QSO: 7.

## Notes

Inspect GALAXY->STAR and STAR<->QSO confusions before scaling. If CNN baselines remain unstable, increase sample size and consider calibrated FITS-based inputs in a later image-only refinement.
