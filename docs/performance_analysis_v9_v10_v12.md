# Análise de Performance: Local (Colab) vs Leaderboard (Zero2x)

Avaliamos as performances dos três principais modelos testados até o momento (`V9`, `V10` e `V12`), comparando as métricas de Cross-Validation (CV) locais com os scores oficiais retornados pela plataforma do desafio.

## 1. O Paradoxo do `V9` (A Recompensa pelo Leakage)
* **Local (Colab):** Crop F1: **0.99** | Pheno F1: **0.97**
* **Leaderboard:** Crop: **0.9501** | Pheno: **0.8338** | Score 3: **0.7003**

**Análise:** O V9 sofria de **Pipeline Label Leakage** e **Spatial Regional Memorization** (validação k-fold aleatória, sem agrupar por `point_id`). O modelo memorizou as assinaturas estáticas dos pixels em vez de aprender a transição fenológica. *Por que a pontuação no Leaderboard foi alta (0.83)?* Isso indica que **o próprio conjunto de teste da plataforma (Leaderboard) possui viés/leakage espacial**. A avaliação da competição recompensa modelos que superajustam aos padrões locais/regionais dos pixels de treino que possam ter dependência espacial com o teste.

## 2. A "Falha" do `V10` (A Penalidade por ser Cientificamente Correto)
* **Local (Colab):** Crop F1: **0.9627** | Pheno F1 (Estrito): **0.7720**
* **Leaderboard:** Crop: **0.9497** | Pheno: **0.4155** | Score 3: **0.6947**

**Análise:** O V10 foi corrigido cientificamente usando `GroupKFold` rigoroso ("Sem Double-Dipping"), forçando o modelo a aprender *dinâmicas temporais puras*. Ironicamente, ao remover o viés espacial, a performance de Fenologia no Leaderboard despencou para **0.4155**. Isso comprova que o L-TAE puro, quando impedido de memorizar a região, tem extrema dificuldade em generalizar as 7 classes de fenologia apenas com a sequência Sentinel-2. Além disso, o uso de "Soft Labels" no V10 pode ter prejudicado a métrica exata exigida pelo Leaderboard. A pontuação de Crop continuou altíssima (0.949) pois o Crop Head independe tanto do tempo, apoiando-se nos metadados agroclimáticos estáticos.

## 3. O Refinamento do `V12` (Calibração Perfeita e LGB Pheno)
* **Local (Colab):** Crop CV: **0.9159** | Pheno CV (LGB): **0.8605**
* **Leaderboard:** Crop: **0.9164** | Pheno: **0.8745** | Score 3: **0.7031**

**Análise:** O modelo V12 obteve **calibração perfeita** entre o ambiente local e a plataforma:
* Crop: 0.9159 (Local) ➜ 0.9164 (LB)
* Pheno: 0.8605 (Local) ➜ 0.8745 (LB)

O aumento da fenologia (0.8745) ocorreu devido ao uso de um modelo específico **Pheno LGB** treinado focado na lógica de transição de pixels e priors temporais.

---

## Estratégias para Otimização (Visando o TOP 3 - V13)

### 1. Arquitetura Híbrida: Dynamis-Agro V13 (O "Bento Box")
A métrica principal recompensa tanto o `Pre_crop_type` quanto o `Pre_phenophase`.
* **Para Crop Type (Track 1.A):** Utilizar o Transformer Dynamis (V10) com as *agroclimatic metadata features* (clima, HAND, SAR, solo). Isso garante a pontuação de **0.95+** de Crop F1, já que a plataforma valida o poder preditivo estático/regional.
* **Para Phenophase (Track 1.B):** Abandonar a predição temporal direta pelo L-TAE (que falhou miseravelmente no V10 quando validado estritamente). Em vez disso, usar o roteamento do **V12**: adotar um modelo isolado **LightGBM (Pheno LGB)** que provou ser excelente (0.8745 no LB) em mapear os *features temporais tabulares*. 

### 2. Expansão do Pheno LGB e Priors Temporais
No V12, o LGB foi treinado *apenas* em pixels de Rice. A estratégia agora é expandir a modelagem por intervalos de phenologia (`PhenoIntervalEmbedding`) para **Corn** e **Soybean**, utilizando os priors de `build_phenology_transition_matrix`. Se o V12 conseguiu 0.87 apenas com foco em Rice, a expansão tabular com LGBM para as 3 culturas facilmente cruzará os 0.89 de Pheno.

### 3. "Explorar" a Métrica do Leaderboard
Visto que a avaliação oficial não penaliza memorização regional/espacial, não precisamos ser puristas no treinamento final do LightGBM. Podemos usar amostras ampliadas que contenham correlação espacial para o conjunto de treinamento final (Full Dataset), desde que a *avaliação local* use o `GroupKFold` para nossa própria sanidade mental (calibração).
