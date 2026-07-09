# Multiband Quasar Classification

Projeto para identificar automaticamente objetos astronomicos e preparar a estimativa
de redshift fotometrico usando representacoes de imagem e dados tabulares do mesmo
conjunto de objetos.

Este repositorio inicia pela **Fase 1**: benchmark de classificacao por imagem
para distinguir `STAR`, `GALAXY` e `QSO` usando cutouts JPEG do SDSS DR17.

## Fases

1. **Imagem**: benchmark com `simple_cnn`, `resnet18` e `densenet121`.
2. **Tabular**: benchmark futuro usando magnitudes, cores e atributos fotometricos.
3. **Hibrido**: comparacao imagem vs tabular e fusao multimodal.

## Fonte dos dados

A Fase 1 usa SDSS DR17:

- Catalogo: SkyServer SQL Search, consultando fotometria em `PhotoObj` e rotulos
  espectroscopicos em `SpecObj.class`.
- Imagens: SkyServer ImgCutout `getjpeg`, baixando cutouts pelos campos `ra` e `dec`.

Cada linha de `metadata.csv` preserva `object_id`, coordenadas, classe,
redshift espectroscopico, magnitudes, extincoes, cores derivadas, split e caminho
do cutout. Isso permite reaproveitar exatamente os mesmos objetos nas Fases 2 e 3.

O MultimodalUniverse fica documentado como opcao futura de referencia multimodal,
mas nao e dependencia da Fase 1.

## Estrutura

```text
configs/
  image_benchmark.yaml
data/
  raw/catalogs/
  raw/images/{STAR,GALAXY,QSO}/
  processed/image_sample/
notebooks/image_benchmark/
reports/image_benchmark/
runs/image_benchmark/
src/multiband_qso/
tests/
```

## Instalar

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
```

## Validacao inicial da Fase 1

Esta validacao executa apenas a Fase 1: classificacao por imagem com cutouts JPEG do SDSS. A estimativa de redshift fotometrico sera tratada depois, principalmente na Fase 2, e a abordagem hibrida multimodal fica reservada para a Fase 3.

Instalar o projeto e dependencias de desenvolvimento:

```bash
python -m pip install -e ".[dev]"
```

Rodar testes locais e verificacao de sintaxe:

```bash
python -m pytest -q
python -m compileall src tests
```

Executar o smoke test SDSS com amostra pequena:

```bash
sdss-fetch-catalog --config configs/image_benchmark_smoke.yaml
sdss-build-metadata --config configs/image_benchmark_smoke.yaml
sdss-download-cutouts --config configs/image_benchmark_smoke.yaml
sdss-validate-metadata --config configs/image_benchmark_smoke.yaml
```

A configuracao `configs/image_benchmark_smoke.yaml` usa 10 objetos por classe (`STAR`, `GALAXY`, `QSO`), split estratificado 60/20/20, `seed=42`, treino em CPU, `epochs=2`, `batch_size=8` e `num_workers=0`. Para reduzir fragilidade do endpoint SQL durante o smoke test, essa configuracao usa filtros SDSS minimos; a configuracao principal mantem filtros de qualidade mais restritivos.

Treinar somente o baseline minimo `simple_cnn`:

```bash
qso-train-image --config configs/image_benchmark_smoke.yaml --model simple_cnn --run-name smoke-simple-cnn
```

Avaliar o checkpoint gerado:

```bash
qso-evaluate-image --config configs/image_benchmark_smoke.yaml \
  --model simple_cnn \
  --checkpoint runs/image_benchmark/simple_cnn/smoke-simple-cnn/checkpoint_best.pt
```

Gerar o relatorio comparativo:

```bash
qso-image-benchmark-report --runs-dir runs/image_benchmark
```

Artefatos esperados no smoke test:

- `data/processed/image_sample/metadata.csv`
- `data/processed/image_sample/class_mapping.json`
- `data/processed/image_sample/metadata_validation.json`
- `runs/image_benchmark/simple_cnn/smoke-simple-cnn/checkpoint_best.pt`
- `runs/image_benchmark/simple_cnn/smoke-simple-cnn/history.csv`
- `runs/image_benchmark/simple_cnn/smoke-simple-cnn/metrics.json`
- `runs/image_benchmark/simple_cnn/smoke-simple-cnn/metrics_test.json`
- `reports/image_benchmark/model_comparison.md`

O `metadata.csv` preserva identificadores, coordenadas, classe, redshift espectroscopico, magnitudes, extincoes e cores derivadas para reaproveitamento futuro em modelos tabulares e hibridos. Os cutouts JPEG sao adequados para este benchmark inicial de PDI, mas nao substituem FITS calibrados em analises cientificas mais rigorosas.
## Primeira escala experimental da Fase 1

Depois que o smoke test passar, use `configs/image_benchmark_small.yaml` para uma primeira execucao real ainda controlada. Essa configuracao usa cerca de 900 objetos no total: 300 `STAR`, 300 `GALAXY` e 300 `QSO`, com split estratificado 70/15/15.

Diferença entre configs:

- `configs/image_benchmark_smoke.yaml`: validacao funcional minima, 10 objetos por classe, 2 epocas, CPU, usada para confirmar o pipeline de ponta a ponta.
- `configs/image_benchmark_small.yaml`: primeira escala experimental controlada, 300 objetos por classe, 10 epocas, `device: auto`, usada para medir um baseline inicial com `simple_cnn`.

A config pequena escreve em diretorios separados para nao sobrescrever o smoke test:

- `data/processed/image_sample_small/`
- `data/raw/images_small/`

Executar o pipeline pequeno real:

```bash
sdss-fetch-catalog --config configs/image_benchmark_small.yaml
sdss-build-metadata --config configs/image_benchmark_small.yaml
sdss-download-cutouts --config configs/image_benchmark_small.yaml
sdss-validate-metadata --config configs/image_benchmark_small.yaml
```

Treinar e avaliar primeiro apenas o baseline `simple_cnn`:

```bash
qso-train-image --config configs/image_benchmark_small.yaml --model simple_cnn --run-name small-simple-cnn
qso-evaluate-image --config configs/image_benchmark_small.yaml \
  --model simple_cnn \
  --checkpoint runs/image_benchmark/simple_cnn/small-simple-cnn/checkpoint_best.pt
qso-image-benchmark-report --runs-dir runs/image_benchmark
```

Interprete esse resultado como benchmark parcial. Antes de treinar `resnet18` e `densenet121`, verifique se a loss caiu, se as metricas passaram de chance level, se existe overfitting evidente e se a matriz de confusao colapsou em uma unica classe. Se o baseline estiver estavel, os proximos comandos serao:

```bash
qso-train-image --config configs/image_benchmark_small.yaml --model resnet18 --run-name small-resnet18
qso-train-image --config configs/image_benchmark_small.yaml --model densenet121 --run-name small-densenet121
```

Esses comandos ainda pertencem apenas a Fase 1. Nao envolvem modelos tabulares, redshift fotometrico nem arquitetura multimodal.
## Pipeline da Fase 1

Buscar candidatos no SDSS:

```bash
sdss-fetch-catalog --config configs/image_benchmark.yaml
```

Construir o metadata balanceado:

```bash
sdss-build-metadata --config configs/image_benchmark.yaml
```

Baixar cutouts:

```bash
sdss-download-cutouts --config configs/image_benchmark.yaml
```

Treinar um modelo:

```bash
qso-train-image --config configs/image_benchmark.yaml --model simple_cnn
```

Avaliar um checkpoint:

```bash
qso-evaluate-image --config configs/image_benchmark.yaml \
  --model simple_cnn \
  --checkpoint runs/image_benchmark/simple_cnn/<run-name-or-timestamp>/checkpoint_best.pt
```

Gerar tabela comparativa:

```bash
qso-image-benchmark-report --runs-dir runs/image_benchmark
```

## Observacoes cientificas

- Os rotulos vem de `SpecObj.class` com filtros de qualidade espectroscopica.
- Os cutouts JPEG sao adequados para um benchmark inicial de PDI, mas nao substituem
  FITS calibrados quando a analise exigir fotometria ou pixels cientificamente calibrados.
- QSO e STAR podem ser visualmente semelhantes em RGB JPEG; essa limitacao deve aparecer
  nos relatorios da Fase 1 e motiva as Fases 2 e 3.
