# EDA: Phenophase Intervals — Análise Exploratória Completa

**Fonte de dados:** `points_train_label.csv` + `data/cache/phenophases.parquet`
**Pontos analisados:** 778 (todos os dados de treino)
**Data da análise:** 2026-05-03

---

## 1. Ordenação Temporal no CSV

O CSV original **não está ordenado** cronologicamente por ponto. Exemplo — Point ID 1:

| Linha Original | Data | Fase | Status |
|---|---|---|---|
| 0 | 2018-06-07 | Greenup | ✅ |
| 1 | 2018-06-30 | MidGreenup | ✅ |
| **2** | **2018-08-06** | **Peak** | ❌ (fora de ordem) |
| **3** | **2018-07-22** | **Maturity** | ❌ |
| 4 | 2018-09-12 | MidSenescence | ✅ |
| **5** | **2018-08-21** | **Senescence** | ❌ |
| 6 | 2018-10-03 | Dormancy | ✅ |

**Impacto:** Os modelos V9/V10 usam `gaussian_soft_labels` que mapeia cada label ao DOY, sendo imune à desordem das linhas. Sem impacto direto no treino.

---

## 2. Sequência Canônica dos Estágios (Descoberta Crítica)

**100% dos 778 pontos** (rice, corn, soybean) seguem, quando ordenados por data:

> **Greenup → MidGreenup → Maturity → Peak → Senescence → MidSenescence → Dormancy**

> [!IMPORTANT]
> "Maturity" precede "Peak" **por definição do dataset**, não é erro.
> Neste contexto, "Maturity" = início do enchimento de grãos (grain-fill onset),
> e "Peak" = máximo de biomassa/NDVI. Não renomear — o gabarito usa esses nomes.

**Bug no `phenology_prior.py` original:** A sequência estava incorreta (`Dormancy → Greenup → MidGreenup → Peak → Maturity → MidSenescence → Senescence`). **Corrigido.**

---

## 3. Intervalos por Cultura (Média ± Desvio Padrão em Dias)

| Transição | Rice (n=367) | Corn (n=229) | Soybean (n=182) |
|---|---|---|---|
| Greenup → MidGreenup | 20.1 ± 1.5 | 22.3 ± 1.5 | 22.5 ± 1.8 |
| MidGreenup → Maturity | 23.8 ± 1.3 | 24.1 ± 1.1 | 23.0 ± 1.5 |
| Maturity → Peak | 16.7 ± 0.8 | 16.7 ± 1.1 | **14.6 ± 0.8** |
| Peak → Senescence | 19.4 ± 1.0 | 19.1 ± 1.5 | **15.4 ± 0.8** |
| Senescence → MidSenescence | **29.9 ± 1.6** | 27.7 ± 2.7 | **21.4 ± 0.9** |
| MidSenescence → Dormancy | **27.7 ± 2.0** | 25.7 ± 2.4 | **20.2 ± 0.9** |

> [!NOTE]
> **Soja tem senescência ~8.5d mais curta** que arroz. Diferença entre culturas é real e significativa.
> **Priors por cultura são necessários no modelo.**

---

## 4. Variação Regional (Mesma Cultura, Regiões Diferentes)

Análise por 49 regiões via **Signal-to-Noise Ratio** (variação entre regiões ÷ variação intra-ponto):

| Crop | Transition | SNR |
|---|---|---|
| corn | Senescence → MidSenescence | 1.01 |
| rice | Greenup → MidGreenup | 1.03 |
| **todo o resto** | **todas** | **< 1.0** |

> [!IMPORTANT]
> Para **SNR < 1.0**, a variação entre regiões é **menor que o ruído natural de cada ponto**.
> Priors por região **não trazem ganho** — os intervalos observados de cada ponto individual
> já capturam essa variação.

**Conclusão:** Usar os intervalos **observados de cada ponto** + prior **por cultura** como fallback.

---

## 5. Bug de Indexação: Cache vs. phenology_prior.py

| Módulo | Índice 0 | Índice 1 | Sequência |
|---|---|---|---|
| `cache_writer.py` (PHENOPHASES_CANON) | Dormancy | Greenup | Alfabética/legada |
| `phenology_prior.py` (**corrigido**) | Greenup | MidGreenup | **Cronológica** ✅ |

Os dois índices são **independentes**: o cache usa o índice para armazenar dados, o modelo usa o nome da fase via `phenophase_name_to_index()`. Sem conflito em runtime — mas merece documentação.

---

## 6. Impacto nos Modelos V9/V10

| Problema | V9 | V10 | V11 (proposto) |
|---|---|---|---|
| Sequência errada no `phenology_prior.py` | ✅ Não usa | ✅ Não usa | ✅ Corrigido |
| Priors globais (não por cultura) | ❌ N/A | ❌ N/A | ✅ Per-crop |
| `phenophase_interval` como feature | ❌ Ausente | ❌ Ausente | ✅ Implementado |
| Intervalo observado por ponto | ❌ Ausente | ❌ Ausente | ✅ Implementado |
| Soft-labels com σ=10d | ✅ Presente | ✅ Presente | ✅ Mantido |

---

## 7. Arquivos Modificados

- `src/dynamis/phenology_prior.py` — Sequência corrigida, priors por cultura adicionados, `get_crop_interval_prior()` criada
- `src/dynamis/__init__.py` — Novos exports: `PHENOPHASE_INTERVALS_MEAN`, `PHENOPHASE_CUMULATIVE_DAYS`, `get_crop_interval_prior`, `build_phenology_interval_embedding`
- `notebooks/10_dynamis_v11_pheno_interval_ltae.py` — V11 com `PhenoIntervalEmbedding` crop-aware
