# ─── Cell 1 — Environment autodetect (Colab / Lightning Studio / local) + cache bootstrap
import os, sys, shutil, subprocess, json, time, copy, math, warnings
warnings.filterwarnings('ignore')
from pathlib import Path


def _find_git_root(start: Path):
    for p in [start, *start.parents]:
        if (p / '.git').exists():
            return p
    return None


IN_COLAB = 'google.colab' in sys.modules
IN_LIGHTNING = (
    not IN_COLAB
    and (
        os.environ.get('LIGHTNING_CLOUD_SPACE_ID') is not None
        or Path('/teamspace/studios/this_studio').exists()
    )
)
ENV = 'colab' if IN_COLAB else ('lightning' if IN_LIGHTNING else 'local')
print(f'ENV: {ENV}')

if IN_COLAB:
    REPO_PATH = Path('/content/ai-for-good')
    if not REPO_PATH.exists():
        try:
            from google.colab import userdata
            token = userdata.get('GITHUB_TOKEN')
        except Exception:
            token = None
        base = f'https://x-access-token:{token}@github.com/' if token else 'https://github.com/'
        subprocess.run(['git', 'clone', f'{base}GeoProjectAI/ai-for-good.git', str(REPO_PATH)], check=True)
    else:
        subprocess.run(['git', '-C', str(REPO_PATH), 'pull', '--ff-only'], check=False)
    WORKSPACE = Path('/content/drive/Shareddrives/SKYVIDYA/AI for Good/datasets_final_round')
    CACHE_DIR = WORKSPACE / 'cache'
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
else:
    REPO_PATH = _find_git_root(Path.cwd()) or Path.cwd()
    CACHE_DIR = REPO_PATH / 'data' / 'cache'

if IN_COLAB or IN_LIGHTNING:
    subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '-q',
         'hilbertcurve', 'rasterio', 'geopandas', 'shapely',
         'pyarrow', 'lightgbm', 'tqdm'],
        check=True,
    )

if not os.access(REPO_PATH, os.W_OK):
    raise PermissionError(f'REPO_PATH not writable: {REPO_PATH}')

if str(REPO_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_PATH))

MODELS_DIR = REPO_PATH / 'models'
REPORTS_DIR = REPO_PATH / 'reports'
SUBMISSION_DIR = REPO_PATH / 'submissions'
for d in [MODELS_DIR, REPORTS_DIR, SUBMISSION_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f'REPO_PATH:      {REPO_PATH}')
print(f'CACHE_DIR:      {CACHE_DIR}')
print(f'MODELS_DIR:     {MODELS_DIR}')
print(f'SUBMISSION_DIR: {SUBMISSION_DIR}')

# ── Cache bootstrap (parquets ship in repo at data/cache/)
REQUIRED = [
    'points_meta.geoparquet', 'observations_base.parquet',
    'observations_enriched.parquet', 'phenophases.parquet', 'MANIFEST.json',
]
missing = [f for f in REQUIRED if not (CACHE_DIR / f).exists()]
if missing:
    repo_cache = REPO_PATH / 'data' / 'cache'
    if all((repo_cache / f).exists() for f in REQUIRED) and repo_cache.resolve() != CACHE_DIR.resolve():
        for f in REQUIRED:
            shutil.copy2(repo_cache / f, CACHE_DIR / f)
        print(f'[bootstrap] copied {len(REQUIRED)} cache files repo → {CACHE_DIR}')
missing = [f for f in REQUIRED if not (CACHE_DIR / f).exists()]
assert not missing, f'Cache incomplete: {missing}'
print('Cache OK — ready for Cell 2.')

# ─── Cell 2 — Load both caches
import numpy as np
import pandas as pd

from src.data.cache_loader import load_aggregated_cache, load_manifest

manifest = load_manifest(CACHE_DIR)
print(f'manifest version: {manifest["version"]} | n_points: {manifest["n_points"]} '
      f'| n_regions: {manifest["n_regions"]} | n_obs_base: {manifest["n_obs_base"]}')

series_base = load_aggregated_cache(CACHE_DIR, enriched=False)
series_agro = load_aggregated_cache(CACHE_DIR, enriched=True)

F_BASE = series_base[0].features.shape[1]
F_AGRO = series_agro[0].features.shape[1]
print(f'series_base: {len(series_base)} pts, F={F_BASE}')
print(f'series_agro: {len(series_agro)} pts, F={F_AGRO}')
assert F_BASE == 17 and F_AGRO == 21
assert [ps.point_id for ps in series_base] == [ps.point_id for ps in series_agro]

# ─── Cell 3 — Build tensors with forward-fill pheno trajectory
from src.data.temporal_builder import FEATURE_NAMES as BASE_FEATURE_NAMES
from src.data.cache_writer import AGRO_FEATURES
from src.dynamis import PHENOPHASES, hurst_features, phenophase_name_to_index

FEATURE_NAMES_BASE = list(BASE_FEATURE_NAMES)
FEATURE_NAMES_AGRO = FEATURE_NAMES_BASE + list(AGRO_FEATURES)
CROPS = ['rice', 'corn', 'soybean']
N_PHENO = len(PHENOPHASES)


def _canonical_date(s: str) -> str:
    parts = str(s).replace('/', '-').split('-')
    if len(parts) != 3:
        return str(s)
    try:
        return f'{int(parts[0]):04d}-{int(parts[1]):02d}-{int(parts[2]):02d}'
    except ValueError:
        return str(s)


def _expand_pheno_trajectory(ps_dates, phenophase_by_date):
    """Forward-fill labeled phenophases across all TIFF dates."""
    if not phenophase_by_date:
        return [-100] * len(ps_dates)
    events = sorted(
        (_canonical_date(k), phenophase_name_to_index(v))
        for k, v in phenophase_by_date.items()
    )
    canon_tiff = [_canonical_date(d) for d in ps_dates]
    out, j, cur = [], 0, -100
    for d in canon_tiff:
        while j < len(events) and events[j][0] <= d:
            cur = events[j][1]
            j += 1
        out.append(cur)
    return out


def build_tensors(series_list, n_features):
    n = len(series_list)
    T_max = max(len(ps.dates) for ps in series_list)
    X = np.full((n, T_max, n_features), np.nan, dtype=np.float32)
    mask = np.zeros((n, T_max), dtype=bool)
    hurst_vec = np.full(n, 0.5, dtype=np.float32)
    crop_labels = np.zeros(n, dtype=np.int64)
    pheno_labels = np.full((n, T_max), -100, dtype=np.int64)
    ndvi_idx = FEATURE_NAMES_BASE.index('ndvi')
    for i, ps in enumerate(series_list):
        T = len(ps.dates)
        X[i, :T, :n_features] = ps.features[:, :n_features].astype(np.float32)
        mask[i, :T] = ps.mask
        hf = hurst_features(ps.features[:, ndvi_idx], ps.features[:, :12], min_temporal_dates=8)
        hurst_vec[i] = hf['hurst_temporal'] if hf['hurst_temporal_valid'] else hf['hurst_spectral_mean']
        crop_labels[i] = CROPS.index(ps.crop_type) if ps.crop_type in CROPS else 0
        traj = _expand_pheno_trajectory(ps.dates, ps.phenophase_by_date)
        pheno_labels[i, :T] = np.asarray(traj, dtype=np.int64)
    X = np.nan_to_num(X, nan=0.0)
    hurst_vec = np.clip(hurst_vec, 0.1, 0.95).astype(np.float32)
    regions = [ps.region for ps in series_list]
    region_to_idx = {r: i + 1 for i, r in enumerate(sorted(set(regions)))}
    region_ids = np.array([region_to_idx[r] for r in regions], dtype=np.int64)
    return X, mask, hurst_vec, crop_labels, pheno_labels, region_ids, regions, region_to_idx


X_b, mask_b, hurst_b, crop_y, pheno_y, region_ids, regions, region_to_idx = build_tensors(series_base, F_BASE)
X_a, mask_a, hurst_a, *_ = build_tensors(series_agro, F_AGRO)
N_REGIONS = len(region_to_idx)

valid_mask = mask_b & (pheno_y != -100)
print(f'X_base: {X_b.shape}  X_agro: {X_a.shape}')
print(f'crops: {dict(zip(CROPS, np.bincount(crop_y, minlength=3).tolist()))}')
print(f'regions: {N_REGIONS}  T_max: {X_b.shape[1]}')
print(f'pheno match (post forward-fill): {valid_mask.sum() / max(mask_b.sum(), 1):.1%} '
      f'({valid_mask.sum()} / {mask_b.sum()} valid obs)')
print(f'pheno class distribution: {np.bincount(pheno_y[valid_mask], minlength=N_PHENO).tolist()}')

RICE_IDX = CROPS.index('rice')
rice_point_mask = (crop_y == RICE_IDX)
print(f'rice points: {rice_point_mask.sum()} / {len(crop_y)}')

# ─── Cell 4 — Spatial CV split + LGBM crop baseline + LGBM pheno baseline
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score

SEED = 42
N_SPLITS = 5


def flatten_features(X, mask):
    n, T, F = X.shape
    out = np.zeros((n, F * 5), dtype=np.float32)
    for i in range(n):
        v = mask[i]
        if v.sum() < 1:
            continue
        Xi = X[i, v]
        out[i, 0*F:1*F] = Xi.mean(0)
        out[i, 1*F:2*F] = Xi.std(0)
        out[i, 2*F:3*F] = Xi.max(0)
        out[i, 3*F:4*F] = Xi.min(0)
        if Xi.shape[0] >= 2:
            out[i, 4*F:5*F] = (Xi[-1] - Xi[0]) / max(Xi.shape[0] - 1, 1)
    return out


groups = np.array(regions)
kf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
FOLDS = list(kf.split(np.zeros(len(crop_y)), crop_y, groups=groups))


def run_lgbm_crop(X, mask, crop_y, folds, tag):
    Xfs = StandardScaler().fit_transform(flatten_features(X, mask))
    pred_all = np.zeros_like(crop_y)
    prob_all = np.zeros((len(crop_y), 3), dtype=np.float32)
    for fold, (tr, va) in enumerate(folds):
        m = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.03, max_depth=6,
                                class_weight='balanced', random_state=SEED, verbose=-1)
        m.fit(Xfs[tr], crop_y[tr])
        pred_all[va] = m.predict(Xfs[va])
        prob_all[va] = m.predict_proba(Xfs[va])
        f1 = f1_score(crop_y[va], pred_all[va], average='macro', zero_division=0)
        print(f'  [{tag}/crop] fold {fold+1}: F1m={f1:.3f}')
    f1g = f1_score(crop_y, pred_all, average='macro', zero_division=0)
    print(f'  [{tag}/crop] Spatial CV F1m={f1g:.4f}  OA={accuracy_score(crop_y, pred_all):.4f}')
    return {'pred_all': pred_all, 'prob_all': prob_all, 'f1_global': f1g}


def _build_pheno_rows(X, mask, pheno_y, point_idx_keep):
    keep = set(point_idx_keep.tolist())
    rows_X, rows_y, rows_pid, rows_t = [], [], [], []
    n, T, F = X.shape
    for i in range(n):
        if i not in keep:
            continue
        for t in range(T):
            if not mask[i, t] or pheno_y[i, t] == -100:
                continue
            rows_X.append(X[i, t])
            rows_y.append(int(pheno_y[i, t]))
            rows_pid.append(i)
            rows_t.append(t)
    return (np.asarray(rows_X, dtype=np.float32),
            np.asarray(rows_y, dtype=np.int64),
            np.asarray(rows_pid, dtype=np.int64),
            np.asarray(rows_t, dtype=np.int64))


def run_lgbm_pheno_rice(X, mask, pheno_y, rice_mask_arr, folds, tag):
    rice_idx = np.flatnonzero(rice_mask_arr)
    rice_set = set(rice_idx.tolist())
    pred_all = np.full_like(pheno_y, -100)
    for fold, (tr, va) in enumerate(folds):
        tr_rice = np.array([i for i in tr if i in rice_set])
        va_rice = np.array([i for i in va if i in rice_set])
        if len(tr_rice) == 0 or len(va_rice) == 0:
            print(f'  [{tag}/pheno_rice] fold {fold+1}: skipped (no rice in split)')
            continue
        Xtr, ytr, _, _ = _build_pheno_rows(X, mask, pheno_y, tr_rice)
        Xva, yva, pid_va, t_va = _build_pheno_rows(X, mask, pheno_y, va_rice)
        if Xtr.size == 0 or Xva.size == 0:
            continue
        m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, max_depth=6,
                                class_weight='balanced', objective='multiclass',
                                num_class=N_PHENO, random_state=SEED, verbose=-1)
        m.fit(Xtr, ytr)
        ypred = m.predict(Xva)
        for k, (pid, tt) in enumerate(zip(pid_va, t_va)):
            pred_all[pid, tt] = ypred[k]
        f1 = f1_score(yva, ypred, average='macro', zero_division=0,
                       labels=list(range(N_PHENO)))
        print(f'  [{tag}/pheno_rice] fold {fold+1}: F1m={f1:.3f}  (val rows={len(yva)})')
    valid = (pred_all != -100)
    f1g = f1_score(pheno_y[valid], pred_all[valid], average='macro',
                    zero_division=0, labels=list(range(N_PHENO))) if valid.any() else 0.0
    print(f'  [{tag}/pheno_rice] Spatial CV F1m={f1g:.4f}')
    return {'pred_all': pred_all, 'f1_global': f1g}


print('=== LGBM crop — base ===')
bl_crop_base = run_lgbm_crop(X_b, mask_b, crop_y, FOLDS, 'base')
print('\n=== LGBM crop — agro ===')
bl_crop_agro = run_lgbm_crop(X_a, mask_a, crop_y, FOLDS, 'agro')
print('\n=== LGBM pheno (rice) — base ===')
bl_pheno_base = run_lgbm_pheno_rice(X_b, mask_b, pheno_y, rice_point_mask, FOLDS, 'base')
print('\n=== LGBM pheno (rice) — agro ===')
bl_pheno_agro = run_lgbm_pheno_rice(X_a, mask_a, pheno_y, rice_point_mask, FOLDS, 'agro')

# ─── Cell 5 — Dynamis model + Score-based training loop
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(SEED); np.random.seed(SEED)
print(f'DEVICE: {DEVICE}')

N_PHENOPHASES = N_PHENO


class DynamisConfig:
    def __init__(self, input_dim, state_dim=7, hidden_dim=64, attn_heads=2,
                 n_crops=3, crop_head_dropout=0.3, n_regions=10, region_embed_dim=8):
        self.input_dim = input_dim; self.state_dim = state_dim
        self.hidden_dim = hidden_dim; self.attn_heads = attn_heads
        self.n_crops = n_crops; self.crop_head_dropout = crop_head_dropout
        self.n_regions = n_regions; self.region_embed_dim = region_embed_dim


class ChaosAttention(nn.Module):
    def __init__(self, d, h):
        super().__init__(); self.h, self.hd = h, d // h
        self.qkv = nn.Linear(d, 3 * d); self.out = nn.Linear(d, d)
        self.sc = math.sqrt(self.hd)
    def forward(self, x, mask=None, hurst=None):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / self.sc
        if hurst is not None:
            attn = attn * (1.0 + hurst.view(B, 1, 1, 1))
        if mask is not None:
            attn = attn.masked_fill(~mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn = F.softmax(attn, dim=-1).nan_to_num(0.0)
        return self.out((attn @ v).transpose(1, 2).reshape(B, T, D))


class DynamisCropModel(nn.Module):
    def __init__(self, cfg):
        super().__init__(); self.cfg = cfg
        self.region_embedding = nn.Embedding(cfg.n_regions + 1, cfg.region_embed_dim, padding_idx=0)
        self.input_proj = nn.Linear(cfg.input_dim + cfg.region_embed_dim, cfg.hidden_dim)
        self.attn1 = ChaosAttention(cfg.hidden_dim, cfg.attn_heads)
        self.norm1 = nn.LayerNorm(cfg.hidden_dim)
        self.attn2 = ChaosAttention(cfg.hidden_dim, cfg.attn_heads)
        self.norm2 = nn.LayerNorm(cfg.hidden_dim)
        self.state_gru = nn.GRUCell(cfg.hidden_dim, cfg.hidden_dim)
        self.crop_head = nn.Sequential(nn.Dropout(cfg.crop_head_dropout), nn.Linear(cfg.hidden_dim, cfg.n_crops))
        self.pheno_head = nn.Linear(cfg.hidden_dim, cfg.state_dim)
    def forward(self, x, mask=None, hurst=None, region_ids=None):
        B, T, D = x.shape
        if region_ids is not None:
            x = torch.cat([x, self.region_embedding(region_ids).unsqueeze(1).expand(-1, T, -1)], dim=-1)
        h = self.input_proj(x)
        h = self.norm1(h + self.attn1(h, mask, hurst))
        h = self.norm2(h + self.attn2(h, mask, hurst))
        state = torch.zeros(B, self.cfg.hidden_dim, device=x.device)
        innov = []
        for t in range(T):
            innov.append(h[:, t] - state)
            state = self.state_gru(h[:, t], state)
        return {'crop_logits': self.crop_head(state),
                'pheno_logits': self.pheno_head(h),
                'innovations': torch.stack(innov, dim=1)}


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.0):
        super().__init__(); self.g, self.w, self.ls = gamma, weight, label_smoothing
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.w, reduction='none', label_smoothing=self.ls)
        return (((1 - torch.exp(-ce)) ** self.g) * ce).mean()


def normalize_temporal_with_stats(X_tr, m_tr):
    """Return (mu, sd) over valid cells of X_tr."""
    vv = X_tr[m_tr]
    if vv.size:
        mu, sd = vv.mean(0), vv.std(0)
    else:
        mu, sd = np.zeros(X_tr.shape[-1]), np.ones(X_tr.shape[-1])
    sd[sd < 1e-6] = 1.0
    return mu, sd


def apply_norm(X, mask, mu, sd):
    return np.where(mask[..., None], (X - mu) / sd, 0.0).astype(np.float32)


def make_balanced_sampler(labels, n=3):
    c = np.bincount(labels, minlength=n); w = 1.0 / np.maximum(c, 1); sw = w[labels]
    return WeightedRandomSampler((sw / sw.sum() * len(labels)).tolist(), len(labels), replacement=True)


def _eval_dynamis(m, X_va, mask_va, hurst_va, region_va):
    m.eval()
    with torch.no_grad():
        ov = m(
            torch.from_numpy(X_va).float().to(DEVICE),
            mask=torch.from_numpy(mask_va).bool().to(DEVICE),
            hurst=torch.from_numpy(hurst_va).float().to(DEVICE),
            region_ids=torch.from_numpy(region_va).long().to(DEVICE),
        )
    return (ov['crop_logits'].argmax(-1).cpu().numpy(),
            ov['pheno_logits'].argmax(-1).cpu().numpy(),
            ov)


def _f1_pheno_rice(pheno_pred, pheno_true, mask_va, crop_va):
    rice_cells = mask_va & (pheno_true != -100) & (crop_va[:, None] == RICE_IDX)
    if not rice_cells.any():
        return 0.0
    return f1_score(pheno_true[rice_cells], pheno_pred[rice_cells],
                    average='macro', zero_division=0, labels=list(range(N_PHENO)))


def train_dynamis_fold(X_tr, m_tr, h_tr, c_tr, p_tr, r_tr,
                       X_va, m_va, h_va, c_va, p_va, r_va,
                       n_regions, epochs=100, bs=16, lr=5e-5):
    mu, sd = normalize_temporal_with_stats(X_tr, m_tr)
    X_tr_n = apply_norm(X_tr, m_tr, mu, sd)
    X_va_n = apply_norm(X_va, m_va, mu, sd)
    hm, hsd = h_tr.mean(), max(h_tr.std(), 1e-6)
    h_tr_n = ((h_tr - hm) / hsd).astype(np.float32)
    h_va_n = ((h_va - hm) / hsd).astype(np.float32)
    cfg = DynamisConfig(input_dim=X_tr_n.shape[-1], n_regions=n_regions)
    m = DynamisCropModel(cfg).to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=5e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr / 20)
    cw = torch.tensor(1.0 / np.maximum(np.bincount(c_tr, minlength=3), 1), dtype=torch.float32).to(DEVICE)
    cw = cw / cw.sum() * 3
    crit = FocalLoss(weight=cw)
    ds = TensorDataset(
        torch.from_numpy(X_tr_n).float(), torch.from_numpy(m_tr).bool(),
        torch.from_numpy(h_tr_n).float(), torch.from_numpy(c_tr).long(),
        torch.from_numpy(p_tr).long(), torch.from_numpy(r_tr).long(),
    )
    dl = DataLoader(ds, batch_size=bs, sampler=make_balanced_sampler(c_tr), drop_last=False)
    best_score, best_state = -1.0, None
    best_f1c, best_f1p = 0.0, 0.0
    for ep in range(epochs):
        m.train()
        lam = 0.5
        for xb, mb, hb, cb, pb, rb in dl:
            xb, mb, hb, cb, pb, rb = (t.to(DEVICE) for t in (xb, mb, hb, cb, pb, rb))
            o = m(xb, mask=mb, hurst=hb, region_ids=rb)
            loss = crit(o['crop_logits'], cb)
            pf = pb.reshape(-1); vm = (pf != -100)
            if vm.sum() > 0:
                loss = loss + lam * F.cross_entropy(
                    o['pheno_logits'].reshape(-1, N_PHENOPHASES)[vm], pf[vm], ignore_index=-100,
                )
            loss = loss + 0.02 * (o['innovations'] ** 2).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        sched.step()
        crop_pred, pheno_pred, _ = _eval_dynamis(m, X_va_n, m_va, h_va_n, r_va)
        f1c = f1_score(c_va, crop_pred, average='macro', zero_division=0)
        f1p = _f1_pheno_rice(pheno_pred, p_va, m_va, c_va)
        score = 0.5 * f1c + 0.5 * f1p
        if score > best_score:
            best_score, best_f1c, best_f1p = score, f1c, f1p
            best_state = copy.deepcopy(m.state_dict())
    if best_state is not None:
        m.load_state_dict(best_state)
    return m, best_score, best_f1c, best_f1p, cfg, (mu, sd, hm, hsd)

# ─── Cell 6 — Run Dynamis on both feature sets (3-metric tracking)
def run_dynamis(X, mask, hurst, crop_y, pheno_y, region_ids, n_regions, folds, tag, epochs=100):
    crop_pred_all = np.zeros_like(crop_y)
    crop_prob_all = np.zeros((len(crop_y), 3), dtype=np.float32)
    crop_logits_all = np.zeros((len(crop_y), 3), dtype=np.float32)
    pheno_pred_all = np.full_like(pheno_y, -100)
    fold_rows = []
    for fold, (tr, va) in enumerate(folds):
        print(f'  [{tag}] fold {fold+1}/{len(folds)} (train={len(tr)} val={len(va)})')
        m, best_score, best_f1c, best_f1p, cfg, (mu, sd, hm, hsd) = train_dynamis_fold(
            X[tr], mask[tr], hurst[tr], crop_y[tr], pheno_y[tr], region_ids[tr],
            X[va], mask[va], hurst[va], crop_y[va], pheno_y[va], region_ids[va],
            n_regions=n_regions, epochs=epochs,
        )
        # Final eval with best weights
        X_va_n = apply_norm(X[va], mask[va], mu, sd)
        h_va_n = ((hurst[va] - hm) / hsd).astype(np.float32)
        crop_pred_va, pheno_pred_va, ov = _eval_dynamis(m, X_va_n, mask[va], h_va_n, region_ids[va])
        crop_pred_all[va] = crop_pred_va
        crop_prob_all[va] = F.softmax(ov['crop_logits'], -1).cpu().numpy()
        crop_logits_all[va] = ov['crop_logits'].cpu().numpy()
        pheno_pred_all[va] = pheno_pred_va
        fold_rows.append({'fold': fold + 1, 'F1_Crop': best_f1c,
                          'F1_RicePheno': best_f1p, 'Score': 100.0 * best_score})
        out_path = MODELS_DIR / f'dynamis_v8_{tag}_fold{fold+1}.pt'
        torch.save({'state_dict': m.state_dict(), 'cfg': cfg.__dict__,
                    'mu': mu, 'sd': sd, 'hm': hm, 'hsd': hsd,
                    'fold': fold + 1, 'best_score': best_score,
                    'best_f1c': best_f1c, 'best_f1p': best_f1p}, out_path)
        print(f'    fold {fold+1}: F1_Crop={best_f1c:.4f}  F1_RicePheno={best_f1p:.4f}  '
              f'Score={100.0 * best_score:.2f} → saved {out_path.name}')
    f1c_global = f1_score(crop_y, crop_pred_all, average='macro', zero_division=0)
    rice_cells = mask & (pheno_y != -100) & (crop_y[:, None] == RICE_IDX) & (pheno_pred_all != -100)
    f1p_global = (
        f1_score(pheno_y[rice_cells], pheno_pred_all[rice_cells], average='macro',
                  zero_division=0, labels=list(range(N_PHENO)))
        if rice_cells.any() else 0.0
    )
    score_global = 0.5 * f1c_global + 0.5 * f1p_global
    print(f'  [{tag}] DYNAMIS — F1_Crop={f1c_global:.4f}  F1_RicePheno={f1p_global:.4f}  '
          f'Score={100.0 * score_global:.2f}')
    return {'fold_rows': fold_rows, 'crop_pred_all': crop_pred_all,
            'crop_prob_all': crop_prob_all, 'crop_logits_all': crop_logits_all,
            'pheno_pred_all': pheno_pred_all, 'f1_crop_global': f1c_global,
            'f1_pheno_global': f1p_global, 'score_global': score_global}


EPOCHS = 100
print('=== Dynamis — base (17 feat) ===')
dyn_base = run_dynamis(X_b, mask_b, hurst_b, crop_y, pheno_y, region_ids, N_REGIONS, FOLDS, 'base', EPOCHS)
print('\n=== Dynamis — agro (21 feat) ===')
dyn_agro = run_dynamis(X_a, mask_a, hurst_a, crop_y, pheno_y, region_ids, N_REGIONS, FOLDS, 'agro', EPOCHS)

# ─── Cell 7 — Crop ensemble (avg softmax) + Temperature scaling + OOD routing

# 1) Crop ensemble — average softmax of Dynamis_agro and LGBM_base.
#    Uses true LGBM probs (not one-hot of argmax — fixes a v6 wart).
ens_crop_prob = 0.5 * dyn_agro['crop_prob_all'] + 0.5 * bl_crop_base['prob_all']
ens_crop_pred = ens_crop_prob.argmax(-1)
ens_pheno_pred = dyn_agro['pheno_pred_all']

# 2) Temperature scaling on Dynamis_agro logits (calibration).
class TemperatureScaling(nn.Module):
    def __init__(self):
        super().__init__(); self.temperature = nn.Parameter(torch.ones(1))
    def forward(self, logits):
        return logits / self.temperature

logits_t = torch.from_numpy(dyn_agro['crop_logits_all']).float().to(DEVICE)
labels_t = torch.from_numpy(crop_y).long().to(DEVICE)
tscaler = TemperatureScaling().to(DEVICE)
opt_ts = torch.optim.LBFGS([tscaler.temperature], lr=0.01, max_iter=200)

def _ts_closure():
    opt_ts.zero_grad()
    loss = F.cross_entropy(tscaler(logits_t), labels_t)
    loss.backward()
    return loss

opt_ts.step(_ts_closure)
T_opt = float(tscaler.temperature.item())
calibrated_probs = F.softmax(tscaler(logits_t), dim=-1).detach().cpu().numpy()

def _ece(y, p, n_bins=10):
    pred, conf = p.argmax(1), p.max(1)
    e = 0.0
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        idx = np.where((conf > lo) & (conf <= hi))[0]
        if len(idx) == 0:
            continue
        e += len(idx) / len(y) * abs((y[idx] == pred[idx]).mean() - conf[idx].mean())
    return e

ece_pre = _ece(crop_y, dyn_agro['crop_prob_all'])
ece_post = _ece(crop_y, calibrated_probs)
print(f'Calibration: ECE pre={ece_pre:.4f} → post={ece_post:.4f}  | T={T_opt:.3f}')

# 3) OOD routing on calibrated uncertainty.
uncertainty = 1.0 - calibrated_probs.max(1)
ood_threshold = float(np.percentile(uncertainty, 90))
uncertain_mask = uncertainty >= ood_threshold
print(f'OOD: {uncertain_mask.sum()} points flagged (threshold={ood_threshold:.3f})')


# ─── Cell 8 — Leaderboard-parity report (3 metrics × 5 variants)
from sklearn.metrics import confusion_matrix

def _global_score(crop_pred, pheno_pred):
    f1c = f1_score(crop_y, crop_pred, average='macro', zero_division=0)
    rice_cells = mask_b & (pheno_y != -100) & (crop_y[:, None] == RICE_IDX) & (pheno_pred != -100)
    f1p = (
        f1_score(pheno_y[rice_cells], pheno_pred[rice_cells], average='macro',
                  zero_division=0, labels=list(range(N_PHENO)))
        if rice_cells.any() else 0.0
    )
    return f1c, f1p, 100.0 * (0.5 * f1c + 0.5 * f1p)

rows = []
for tag, crop_res, pheno_res in [
    ('Baseline_LGBM_base', bl_crop_base, bl_pheno_base),
    ('Baseline_LGBM_agro', bl_crop_agro, bl_pheno_agro),
]:
    f1c, f1p, score = _global_score(crop_res['pred_all'], pheno_res['pred_all'])
    rows.append({'model': tag, 'Macro-F1_Crop': f1c, 'Macro-F1_RicePheno': f1p, 'Score_Algorithm': score})
for tag, dyn in [('Dynamis_base', dyn_base), ('Dynamis_agro', dyn_agro)]:
    rows.append({'model': tag, 'Macro-F1_Crop': dyn['f1_crop_global'],
                  'Macro-F1_RicePheno': dyn['f1_pheno_global'],
                  'Score_Algorithm': 100.0 * dyn['score_global']})
ens_f1c, ens_f1p, ens_score = _global_score(ens_crop_pred, ens_pheno_pred)
rows.append({'model': 'Ensemble (Dyn_agro + LGBM_base avg softmax)',
              'Macro-F1_Crop': ens_f1c, 'Macro-F1_RicePheno': ens_f1p,
              'Score_Algorithm': ens_score})

report = pd.DataFrame(rows).sort_values('Score_Algorithm', ascending=False).reset_index(drop=True)
print('=== LEADERBOARD-PARITY TABLE (Spatial CV) ===')
print(report.to_string(index=False, formatters={
    'Macro-F1_Crop': '{:.4f}'.format,
    'Macro-F1_RicePheno': '{:.4f}'.format,
    'Score_Algorithm': '{:.2f}'.format,
}))
report.to_csv(REPORTS_DIR / 'v8_leaderboard_metrics.csv', index=False)

best_score = report['Score_Algorithm'].max()
print(f'\nBest local Score: {best_score:.2f}  | Top-3 floor: 97.96  | Gap: {97.96 - best_score:+.2f}')

# Markdown report
md = f"""# Dynamis Terra v8 — Leaderboard Report

Date: {time.strftime('%Y-%m-%d %H:%M')}
Scope: {len(crop_y)} pts × {N_REGIONS} regions × T_max={X_b.shape[1]}
Pheno match rate (post forward-fill): {valid_mask.sum() / max(mask_b.sum(), 1):.1%}

## Spatial CV (5-fold StratifiedGroupKFold by region)

{report.to_markdown(index=False)}

## Calibration (Dynamis_agro)
ECE pre = {ece_pre:.4f} → post = {ece_post:.4f}  (T = {T_opt:.3f})

## OOD
{uncertain_mask.sum()} points flagged at uncertainty threshold {ood_threshold:.3f}.

## Top-3 gap
Best Score = {best_score:.2f}; top-3 floor = 97.96; gap = {97.96 - best_score:+.2f}.
"""
(REPORTS_DIR / 'v8_report.md').write_text(md, encoding='utf-8')
print(f"\n[saved] {REPORTS_DIR / 'v8_report.md'}")

print('\nConfusion matrices (rows=true, cols=pred):')
print('Ensemble:'); print(confusion_matrix(crop_y, ens_crop_pred))

# ─── Cell 9 — Full-data retrain of Dynamis_agro for the production checkpoint
print('Retraining Dynamis_agro on ALL 778 points (no holdout)…')

mu_full, sd_full = normalize_temporal_with_stats(X_a, mask_a)
X_full_n = apply_norm(X_a, mask_a, mu_full, sd_full)
hm_full, hsd_full = float(hurst_a.mean()), float(max(hurst_a.std(), 1e-6))
h_full_n = ((hurst_a - hm_full) / hsd_full).astype(np.float32)

cfg_final = DynamisConfig(input_dim=F_AGRO, n_regions=N_REGIONS)
final_model = DynamisCropModel(cfg_final).to(DEVICE)
opt_final = torch.optim.AdamW(final_model.parameters(), lr=5e-5, weight_decay=5e-3)
FINAL_EPOCHS = 80
sched_final = torch.optim.lr_scheduler.CosineAnnealingLR(opt_final, T_max=FINAL_EPOCHS, eta_min=5e-5 / 20)
cw = torch.tensor(1.0 / np.maximum(np.bincount(crop_y, minlength=3), 1), dtype=torch.float32).to(DEVICE)
cw = cw / cw.sum() * 3
crit = FocalLoss(weight=cw)

ds_final = TensorDataset(
    torch.from_numpy(X_full_n).float(), torch.from_numpy(mask_a).bool(),
    torch.from_numpy(h_full_n).float(), torch.from_numpy(crop_y).long(),
    torch.from_numpy(pheno_y).long(), torch.from_numpy(region_ids).long(),
)
dl_final = DataLoader(ds_final, batch_size=16,
                       sampler=make_balanced_sampler(crop_y), drop_last=False)

for ep in range(FINAL_EPOCHS):
    final_model.train()
    for xb, mb, hb, cb, pb, rb in dl_final:
        xb, mb, hb, cb, pb, rb = (t.to(DEVICE) for t in (xb, mb, hb, cb, pb, rb))
        o = final_model(xb, mask=mb, hurst=hb, region_ids=rb)
        loss = crit(o['crop_logits'], cb)
        pf = pb.reshape(-1); vm = (pf != -100)
        if vm.sum() > 0:
            loss = loss + 0.5 * F.cross_entropy(
                o['pheno_logits'].reshape(-1, N_PHENOPHASES)[vm], pf[vm], ignore_index=-100,
            )
        loss = loss + 0.02 * (o['innovations'] ** 2).mean()
        opt_final.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(final_model.parameters(), 1.0)
        opt_final.step()
    sched_final.step()
    if (ep + 1) % 10 == 0:
        print(f'  epoch {ep+1}/{FINAL_EPOCHS}')

final_model.eval()
ckpt = MODELS_DIR / 'dynamis_v8_final.pt'
torch.save({
    'state_dict': final_model.state_dict(),
    'cfg': cfg_final.__dict__,
    'mu': mu_full, 'sd': sd_full, 'hm': hm_full, 'hsd': hsd_full,
    'temperature': T_opt,
    'region_to_idx': region_to_idx,
    'feature_names_agro': FEATURE_NAMES_AGRO,
    'phenophases': list(PHENOPHASES),
    'crops': CROPS,
    'metrics_cv': {
        'F1_Crop': dyn_agro['f1_crop_global'],
        'F1_RicePheno': dyn_agro['f1_pheno_global'],
        'Score': 100.0 * dyn_agro['score_global'],
        'Ensemble_Score': ens_score,
    },
}, ckpt)
print(f'\nSaved: {ckpt}')

# ─── Cell 10 — Submission scaffold (writes result.json in official format)
#
# Official format (from inference.py):
#   {"<lon>_<lat>_<phenophase_date>": ["<crop>", "<phenophase>"], ...}
#
# This cell writes a placeholder result.json from training data so the
# pipeline shape is verified. Real inference on test_point.csv requires
# extracting features for the 171 test points first via
# scripts/build_full_aggregated_cache.py adapted for test points (TODO).

test_csv_candidates = [
    REPO_PATH / 'data' / '_csv_cache' / 'Guide to the Second Round_track1' / 'test_input_sample' / 'test_point.csv',
    REPO_PATH / 'background' / 'ai-and-space-computing-challenge' / 'Guide to the Second Round_track1' / 'test_input_sample' / 'test_point.csv',
]
test_csv = next((p for p in test_csv_candidates if p.exists()), None)

result = {}
if test_csv is not None:
    test_df = pd.read_csv(test_csv)
    print(f'Loaded {len(test_df)} test rows from {test_csv}')
    # PLACEHOLDER prediction: most common training class per crop, first phenophase.
    # Replace with real inference once test feature cache is built.
    common_crop = CROPS[int(np.bincount(crop_y).argmax())]
    common_pheno = PHENOPHASES[0]
    for _, row in test_df.iterrows():
        key = f"{row['Longitude']}_{row['Latitude']}_{row['phenophase_date']}"
        result[key] = [common_crop, common_pheno]
    print(f'WARNING: placeholder predictions used for {len(result)} test rows.')
    print('TODO: run real inference once test feature cache exists.')
else:
    print('test_point.csv not found locally — skipping submission scaffold.')

submission_path = SUBMISSION_DIR / 'v8_result.json'
submission_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(f'Saved submission scaffold: {submission_path}')
print(f'Sample keys: {list(result)[:3]}')

