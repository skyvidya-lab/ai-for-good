# Análise do Dataset — Track 1

Relatório da análise do pacote **Track 1 Data** da competição ITU AI/ML in 5G (competition-topic1).

## Fonte dos dados

Pacote distribuído em 4 arquivos `.zip` na raiz do repositório:

| Arquivo | Tamanho compactado | Tamanho descompactado | Nº arquivos |
|---|---:|---:|---:|
| [track1_download_link_2.zip](../track1_download_link_2.zip) | 17.48 GB | 17.55 GB | 2574 |
| [track1_download_link_3.zip](../track1_download_link_3.zip) | 17.49 GB | 17.56 GB | 2556 |
| [track1_download_link_4.zip](../track1_download_link_4.zip) | 17.48 GB | 17.55 GB | 2547 |
| [track1_download_link_5.zip](../track1_download_link_5.zip) | 17.48 GB | 17.55 GB | 2568 |
| **Total** | **~69.93 GB** | **~70.21 GB** | **10 245** |

## Estrutura de pastas

Cada zip expande para uma pasta `region_train_N/` plana (sem subpastas), contendo apenas GeoTIFFs:

```
region_train_1/   ← track1_download_link_5.zip
region_train_2/   ← track1_download_link_4.zip
region_train_3/   ← track1_download_link_3.zip
region_train_4/   ← track1_download_link_2.zip
points_train_label.csv   ← incluído em track1_download_link_5.zip (único CSV)
```

Padrão de nome dos TIFFs:
`regionNN_YYYY-MM-DD-HH-MM_YYYY-MM-DD-HH-MM_Sentinel-2_L2A_BXX_(Raw).tiff`

> Observação: uma minoria de arquivos usa `_` em vez de `-` nos componentes de hora (ex.: `region00_2018-10-11-00_00_...`). Há também 1 arquivo com typo (`region542018-07-23-...`, falta underscore após `region54`). Nada disso impede o parse, mas o ingest precisa ser tolerante.

## Número de regiões

**50 IDs de região únicos** no total (`region00` a `region57`, com lacunas — ex.: 03, 05, 14, 19, 22, 23, 30, 41, 55 não existem).

Cobertura por folder (cada folder tem ~47–49 das 50 regiões):

| Folder | Nº regiões | Faltantes em relação ao universo |
|---|---:|---|
| `region_train_1` | 47 | region01, region33, region42 |
| `region_train_2` | 49 | region38 |
| `region_train_3` | 49 | region38 |
| `region_train_4` | 48 | region38, region55 |

Interpretação: as 4 pastas parecem ser **lotes temporais/de reprocessamento do mesmo conjunto espacial de regiões**, não splits disjuntos.

## Número de datas por região

Varia bastante — de **1 a 15 datas** por (folder, região). Média por folder:

| Folder | Datas/região (média) | Mín | Máx |
|---|---:|---:|---:|
| `region_train_1` | 4.7 | 1 | 12 |
| `region_train_2` | 4.4 | 1 | 15 |
| `region_train_3` | 4.4 | 1 | 12 |
| `region_train_4` | 4.5 | 1 | 11 |

Distribuição detalhada disponível por inspeção direta das listagens dos zips (ver seção "Reprodução").

## Bandas Sentinel-2 incluídas

**13 bandas** (todas as bandas L2A distribuídas como GeoTIFFs separados):

`B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B10, B11, B12`

> Nota: B10 normalmente não é incluída em produtos L2A (é banda de cirrus de L1C). A presença aqui é incomum — convém conferir se B10 é utilizável ou se é um placeholder.

O pipeline atual em [extract_features.py](../extract_features.py) usa apenas 12 bandas (B01–B09, B8A, B11, B12). **B10 está sendo ignorada** — decidir se agrega informação.

## Dados fenológicos auxiliares

**Não há arquivo auxiliar separado** de fenologia nos zips. A informação fenológica vem exclusivamente da coluna `phenophase_name` em [points_train_label.csv](../points_train_label.csv):

```
point_id, Longitude, Latitude, phenophase_date, crop_type, phenophase_name
```

- 5 447 linhas
- 778 `point_id` únicos
- **Cada point_id aparece exatamente 7 vezes**, uma para cada um dos 7 estágios fenológicos (`Dormancy`, `Greenup`, `MidGreenup`, `Peak`, `Maturity`, `MidSenescence`, `Senescence`), cada um com uma `phenophase_date` diferente. Distribuição perfeitamente balanceada: 778 amostras por estágio.

## Ground truth de rice paddies

**Não há máscaras binárias rice/no-rice.** O ground truth é **multiclasse em nível de ponto** (não raster), com 3 classes de `crop_type`:

| Classe | Pontos | Linhas (×7 fenofases) |
|---|---:|---:|
| rice | 367 | 2 569 |
| corn | 229 | 1 603 |
| soybean | 182 | 1 274 |
| **Total** | **778** | **5 447** |

> Observação: o README do repo menciona uma 4ª classe `background`, mas ela **não aparece nos labels de treino**. É provável que `background` exista apenas como saída possível na inferência (pontos fora das 3 culturas).

## Split treino/validação dos organizadores

**Não há split oficial** no pacote. Apenas `points_train_label.csv` — sem arquivos `val`, `test` ou coluna de split.

O [train.py](../train.py) usa `GroupKFold` agrupando por `point_id`, o que é a escolha correta dada essa estrutura (evita vazamento entre fenofases do mesmo ponto).

## Reprodução da análise

Listagens dos zips (sem extrair) em `/tmp/list_track1_download_link_{2,3,4,5}.zip.txt`. Comandos-chave:

```bash
# Regiões únicas
grep -hoE 'region[0-9]+_2' /tmp/list_track1_download_link_*.zip.txt | sed 's/_2$//' | sort -u

# Datas por (folder, região)
grep -hoE 'region[0-9]+_[0-9]{4}-[0-9]{2}-[0-9]{2}' /tmp/list_track1_download_link_*.zip.txt | sort -u

# Distribuição de classes
awk -F, 'NR>1{print $5}' points_train_label.csv | sort | uniq -c
```

## Pontos de atenção

1. **B10 nas TIFFs** — incomum em L2A; validar se tem sinal útil ou descartar explicitamente.
2. **Typo `region542018-...`** — 1 arquivo com naming pattern quebrado; garantir parser tolerante.
3. **Dates por região muito irregulares** (1 a 15) — séries curtas vão gerar features temporais pobres. Considerar filtrar pontos em regiões com < N datas, ou imputar.
4. **4 folders com regiões sobrepostas** — precisa decidir se são tratadas como fontes independentes ou consolidadas antes da extração de features. O pipeline atual só usa `region_train_4`.
5. **Sem split oficial** — manter `GroupKFold` por `point_id`; considerar também stratify por `crop_type` × região para avaliar generalização espacial.

---

**Relatório concluído — 2026-04-16** (prazo 2026-04-17 atendido).
