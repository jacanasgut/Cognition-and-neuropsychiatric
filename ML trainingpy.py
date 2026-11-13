# -*- coding: utf-8 -*-
"""
Elastic Net sobre los tests neuropsicológicos (BrainLat, CN y AD).

Objetivo: identificar qué tests (_total) conservan peso predictivo sobre el dolor crónico y sobre
las adversidades infantiles (amenaza, privación), para decidir cuáles entran a los modelos SEM.

Modelos (mismos 11 tests como predictores penalizados; covariables SIN penalizar):
  M1  severidad (solo CP_1 = 1): promedio de z(intensidad), z(frecuencia), z(nº de regiones),
                                 estandarizados dentro de quienes tienen dolor          -> ElasticNetCV
  M2  CP_1 (sí/no)             : presencia de dolor crónico (toda la muestra)          -> LogisticRegressionCV (elasticnet, saga)
      M1 + M2 = modelo de dos partes (hurdle): ¿hay dolor? y, si hay, ¿cuánto?
  M3  amenaza         : media de ítems binarios de violencia física/psicológica      -> ElasticNetCV
  M4  privación       : media de ítems binarios de carencia material/de cuidados     -> ElasticNetCV

Covariables sin penalizar (residualización, teorema de Frisch-Waugh-Lovell):
  - M1, M3, M4: tests y resultado se residualizan sobre las covariables (MCO) y el Elastic Net se ajusta
    sobre los residuos. Es equivalente exacto a un Elastic Net con las covariables sin penalizar.
  - M2 (logística): los tests se residualizan sobre las covariables y las covariables entran como
    predictores. Aproximación (en logística la equivalencia no es exacta), pero impide que un test
    absorba el efecto de una covariable.
  - La residualización se reajusta dentro de cada pliegue de CV y de cada bootstrap (sin filtración).
  - Línea base = MCO / logística sin penalizar solo con covariables; se reporta R²/AUC incremental.

Datos y convenciones:
  - Solo CN y AD (FTD excluido). -99 -> NaN.
  - Tests orientados "más alto = peor" (se invierten mmse, mint, dígitos, minisea); log1p en faq, gad, gds, npi.
  - Valores fuera de rango o placeholders imputados -> NaN (ver clean_tests).
  - Estandarización (z) una vez, sobre la muestra final de casos completos en exposiciones, covariables
    y los tres indicadores de dolor (misma muestra que los modelos SEM). Los tests faltantes se descartan
    por modelo (N final se imprime).
  - Covariables: edad, sexo (mujer = demo_sex == 1), pack-years, años de educación (cog_ed) y
    diagnóstico AD (por defecto; --sin_dx lo quita, solo como análisis de sensibilidad).
  - Exposiciones: ítems binarizados de la hoja "Variables binarizadas"; media por dominio con >= 50% de ítems.
  - Indicadores de dolor: sin dolor (CP_1 == 0) se codifica 0 (66% de ceros: zero-inflation).

Autocontenido: requiere solo pandas, numpy, scikit-learn, matplotlib, openpyxl. No importa otros scripts.

Uso:
  python enet_tests.py [sin_fp] [--sin_dx] [--data=RUTA.xlsx] [--out=CARPETA]
    sin_fp : excluye minisea_fp_total (mucha pérdida de casos)
    --sin_dx: no ajusta por diagnóstico (sensibilidad)
    --data : ACEs_julio_final.xlsx (hojas "Variables binarizadas" y "data raw")
    --out  : carpeta de salida (por defecto, junto a este script)
  Si no se pasa --data, se usa la variable de entorno ACES_DATA y, si no existe,
  ACEs_julio_final.xlsx en la misma carpeta que este script.

Salidas (carpeta resultados_enet_<variante>[_sin_dx]):
  enet_coeficientes.csv   coeficiente, IC95% bootstrap, frecuencia de selección y rendimiento CV por variable y modelo
  enet_pesos_tests.png    coeficientes de los tests por modelo (oscuro = estable)
"""

import os, sys, warnings
import numpy as np, pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import (ElasticNet, ElasticNetCV, LinearRegression, LogisticRegression,
                                  LogisticRegressionCV)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import r2_score, roc_auc_score

warnings.filterwarnings("ignore")

# =====================================================================
# CONFIGURACIÓN  (rutas por línea de comandos o variable de entorno)
# =====================================================================
DATA_FILENAME = "ACEs_julio_final.xlsx"


def _arg(prefix, default=None):
    return next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith(prefix + "=")), default)


VARIANT = "sin_fp" if "sin_fp" in sys.argv else "all"
WITH_DX = "--sin_dx" not in sys.argv       # diagnóstico AD como covariable por defecto
DIAGNOSES_TO_KEEP = ["CN", "AD"]          # FTD siempre excluido
NBOOT, SEED = 200, 2026
L1 = [0.2, 0.5, 0.8, 0.95]
SELECTION_THRESHOLD = 0.8
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:                          # p. ej. ejecución línea a línea en una consola
    _HERE = os.getcwd()
# Ruta del Excel, en este orden: --data=RUTA.xlsx  >  variable de entorno ACES_DATA  >  DATA_FILENAME junto a este script
DATA_PATH = _arg("--data") or os.environ.get("ACES_DATA") or os.path.join(_HERE, DATA_FILENAME)
OUT =os.path.join(_arg("--out", _HERE), f"resultados_enet_{VARIANT}" + ("" if WITH_DX else "_sin_dx"))

# ---------------- ítems de adversidad infantil (nombres en la hoja "Variables binarizadas") ----------------
ASSET = ["msoc_elec_0", "msoc_wat_0", "msoc_runn_0", "msoc_bath_0", "msoc_line_0", "msoc_ref_0", "msoc_wash_0",
         "msoc_tv_0", "msoc_rad_0", "msoc_comp_0", "msoc_inter_0", "msoc_smrt_0", "msoc_sound_0", "msoc_car_0",
         "msoc_bedr_0", "msoc_ob1_0", "msoc_ob2_0", "msoc_ob3_0", "msoc_ob4_0", "msoc_ob5_0"]
BASIC = ["msoc_fin_0", "msoc_prov_period___1", "msoc_prov_period___2", "msoc_eat_0", "msoc_bal_0", "msoc_bas_0",
         "msoc_care_0"]
PHYS = ["msoc_hit_period___1", "msoc_hit_period___2", "msoc_att_period___1", "msoc_att_period___2",
        "msoc_prop_period___1", "msoc_prop_period___2"]
PSYCH = ["msoc_humi_period___1", "msoc_humi_period___2", "msoc_forb_period___1", "msoc_forb_period___2",
         "msoc_mon_period___1", "msoc_mon_period___2", "msoc_deman_period___1", "msoc_deman_period___2"]
EXPOSURES = {"threat": PHYS + PSYCH,              # amenaza = violencia física + psicológica
             "deprivation": ASSET + BASIC}        # privación = carencia material + de cuidados

# ---------------- dolor ----------------
# Nº de regiones dolorosas = suma de 5 grupos anatómicos con al menos una región marcada
PAIN_REGION_GROUPS = {
    "axial": ["CP_Neck_1", "CP_UpperBack_1", "CP_LowerBack_1", "CP_Chest_1", "CP_Abdomen_1"],
    "upper": ["CP_Shoulders_1", "CP_Arms_1", "CP_Elbows_1", "CP_wrists_1", "CP_Hands_1"],
    "lower": ["CP_Legs_1", "CP_Knees_1", "CP_Ankles_1", "CP_Feet_1"],
    "head": ["CP_Head_1", "CP_Face_1"],
    "other": ["CP_Other_1"]}
CP_FREQUENCY_ORDER = {2: 1, 3: 2, 4: 3, 1: 4}     # recodificación de CP_Frequency a orden creciente

# ---------------- tests: columna -> (signo, transformación) ----------------
# signo -1: puntaje alto = mejor desempeño, se invierte para que alto = peor en todos los tests
TESTS = {
    "mmse_total": (-1, None),
    "cog_mint_total": (-1, None),
    "cog_digits_forward_total": (-1, None),
    "cog_digits_backward_total": (-1, None),
    "minisea_em_total": (-1, None),
    "minisea_fp_total": (-1, None),
    "udsfaq_total": (+1, "log1p"),
    "uls_total": (+1, None),
    "gad_total": (+1, "log1p"),
    "gds_total": (+1, "log1p"),
    "npi_total": (+1, "log1p"),
}


def zs(s):
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd and sd > 0 else s * np.nan


def title(t):
    print("\n" + "=" * 90 + f"\n{t}\n" + "=" * 90)


# =====================================================================
# 1. CARGA Y LIMPIEZA
# =====================================================================
def load():
    """Lee todo de ACEs_julio_final.xlsx:
       - hoja "Variables binarizadas" (encabezado en la fila 2): demografía, dolor, tests e ítems ya binarizados
       - hoja "data raw": solo udsfaq_total, que no está en la hoja binarizada (se une por ID)."""
    f = DATA_PATH
    if not os.path.exists(f):
        raise FileNotFoundError(
            f"No existe el archivo: {f}\n"
            f"Indique la ruta con --data=RUTA.xlsx, con la variable de entorno ACES_DATA, "
            f"o coloque {DATA_FILENAME} junto a este script.")
    hojas = pd.ExcelFile(f).sheet_names
    for h in ("Variables binarizadas", "data raw"):
        if h not in hojas:
            raise ValueError(f"El archivo no tiene la hoja '{h}'. Hojas disponibles: {hojas}")

    u = pd.read_excel(f, sheet_name="Variables binarizadas", header=1)
    u.columns = u.columns.astype(str).str.strip().str.replace(r"\s+", "_", regex=True)
    u = u.replace(-99, np.nan)

    raw = pd.read_excel(f, sheet_name="data raw")
    raw.columns = raw.columns.astype(str).str.strip()
    extra = [c for c in TESTS if c not in u.columns and c in raw.columns]      # hoy: udsfaq_total
    if extra:
        u = u.merge(raw[["ID"] + extra].replace(-99, np.nan), on="ID", how="left")
    falt = [c for c in TESTS if c not in u.columns]
    if falt:
        raise ValueError(f"Tests no encontrados en ninguna hoja: {falt}")

    # Tests a numérico: textos como '2,900,925' (placeholder imputado guardado como texto) -> NaN
    for t in TESTS:
        txt = u[t].notna() & pd.to_numeric(u[t], errors="coerce").isna()
        if txt.any():
            print(f"  {t}: {int(txt.sum())} valores de texto -> NaN ({u.loc[txt, t].unique()[:3].tolist()})")
        u[t] = pd.to_numeric(u[t], errors="coerce")

    u = u.dropna(subset=["CP_1"]).copy()
    n_ftd = int((u["clinical_diagnosis"] == "FTD").sum())
    u = u[u["clinical_diagnosis"].isin(DIAGNOSES_TO_KEEP)].copy()
    title("CARGA Y FILTROS")
    print(f"Archivo: {f}  (udsfaq_total desde 'data raw': {'sí' if extra else 'no'})")
    print(f"N tras CP_1 + CN/AD: {len(u)} (FTD excluidos: {n_ftd})")
    print(u["clinical_diagnosis"].value_counts().to_string())
    print(f"CP_1: con dolor={int((u.CP_1 == 1).sum())}, sin dolor={int((u.CP_1 == 0).sum())}")
    return u


def clean_tests(d):
    """Valores imposibles (punto decimal perdido) o placeholders imputados -> NaN."""
    title("LIMPIEZA DE VALORES CORRUPTOS EN TESTS (-> NaN)")

    def setnan(col, mask, why):
        d.loc[mask, col] = np.nan
        print(f"  {col:<14} {int(mask.sum()):>3} valores -> NaN  ({why})")

    setnan("gds_total", d["gds_total"] > 15, "GDS-15: máx 15")
    setnan("npi_total", d["npi_total"] > 144, "NPI-12: máx 144")
    setnan("udsfaq_total", d["udsfaq_total"] > 30, "FAQ: máx 30")
    for col, val in (("gds_total", 2.900925), ("npi_total", 4.285323)):
        setnan(col, np.isclose(d[col], val, atol=1e-5), f"valor repetido {val} = placeholder imputado")
    return d


# =====================================================================
# 2. CONSTRUCCIÓN DE VARIABLES
# =====================================================================
def build(d):
    # Exposiciones: media de ítems binarios, exigiendo >= 50% de ítems observados
    for name, cols in EXPOSURES.items():
        cols = [c for c in cols if c in d.columns]
        X = d[cols].apply(pd.to_numeric, errors="coerce")
        m = X.mean(axis=1)
        m[X.notna().mean(axis=1) < 0.5] = np.nan
        d[f"{name}_raw"] = m

    # Dolor: sin dolor (CP_1 == 0) = 0 en los tres indicadores
    nopain = d["CP_1"] == 0
    d["pain_int_raw"] = pd.to_numeric(d["CP_Intensity"], errors="coerce")
    d.loc[nopain, "pain_int_raw"] = 0
    d["pain_freq_raw"] = pd.to_numeric(d["CP_Frequency"], errors="coerce").map(CP_FREQUENCY_ORDER)
    d.loc[nopain, "pain_freq_raw"] = 0
    d["pain_reg_raw"] = sum(
        d[[c for c in cols if c in d.columns]].apply(pd.to_numeric, errors="coerce").fillna(0).max(axis=1)
        for cols in PAIN_REGION_GROUPS.values())
    d.loc[nopain, "pain_reg_raw"] = 0

    # Covariables
    d["age_raw"] = pd.to_numeric(d["demo_age"], errors="coerce")
    d["sex_female_raw"] = np.where(d["demo_sex"].isna(), np.nan, (d["demo_sex"] == 1).astype(float))
    d["edu_raw"] = pd.to_numeric(d["cog_ed"], errors="coerce")
    pk = (pd.to_numeric(d["demo_smoke_cpd"], errors="coerce") / 20.0) * pd.to_numeric(d["demo_smoke_years"], errors="coerce")
    pk[pd.to_numeric(d["demo_smoke"], errors="coerce") == 0] = 0.0       # nunca fumó -> 0 pack-years
    d["pack_years_raw"] = pk
    d["dx_AD_raw"] = (d["clinical_diagnosis"] == "AD").astype(float)

    # Tests orientados (alto = peor) y transformados
    for t, (sign, tr) in TESTS.items():
        x = pd.to_numeric(d[t], errors="coerce")
        d[t + "_raw"] = sign * (np.log1p(x) if tr == "log1p" else x)
    return d


def finalize(d, covars):
    """Fija la muestra (exposiciones, covariables e indicadores de dolor completos) y estandariza una vez."""
    need = ["threat_raw", "deprivation_raw", "pain_int_raw", "pain_freq_raw", "pain_reg_raw"] + [c + "_raw" for c in covars]
    n0 = len(d)
    s = d.dropna(subset=need).copy()
    title("MUESTRA DE ANÁLISIS")
    print(f"N = {len(s)} (de {n0}; se pierden {n0 - len(s)} por datos faltantes en exposición/covariables/dolor)")
    for v in ["threat", "deprivation", "pain_int", "pain_freq", "pain_reg"] + covars:
        s[v + "_z"] = zs(s[v + "_raw"])
    for t in TESTS:
        s[t + "_z"] = zs(s[t + "_raw"])
    return s


def prep():
    covars = ["age", "sex_female", "pack_years", "edu"] + (["dx_AD"] if WITH_DX else [])
    s = finalize(build(clean_tests(load())), covars)
    tests = [t for t in TESTS if not (VARIANT == "sin_fp" and t == "minisea_fp_total")]
    cols, cv = [t + "_z" for t in tests], [c + "_z" for c in covars]
    s = s.dropna(subset=cols + cv).copy()                      # casos completos en tests
    s["cp1"] = s["CP_1"].astype(int)

    # Parte 2 del hurdle: severidad solo en quienes tienen dolor, con z calculado dentro de ese grupo
    # (en toda la muestra, el 66% de ceros hacía que el compuesto midiera sobre todo "hay / no hay dolor")
    pain = s["cp1"] == 1
    s["pain_sev"] = np.nan
    s.loc[pain, "pain_sev"] = pd.concat([zs(s.loc[pain, v + "_raw"])
                                         for v in ["pain_int", "pain_freq", "pain_reg"]], axis=1).mean(axis=1)
    print(f"\nN para Elastic Net = {len(s)} (con dolor: {int(pain.sum())}) | tests = {len(tests)} | covariables = {covars}")
    return s, tests, cols, cv


# =====================================================================
# 3. MODELOS
# =====================================================================
def fit_reg(X, y, seed):
    kf = KFold(5, shuffle=True, random_state=seed)
    return ElasticNetCV(l1_ratio=L1, cv=kf, alphas=60, max_iter=20000, random_state=seed).fit(X, y)


def fit_clf(X, y, seed):
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    return LogisticRegressionCV(penalty="elasticnet", solver="saga", l1_ratios=L1, Cs=12, cv=skf,
                                scoring="roc_auc", max_iter=5000, random_state=seed).fit(X, y)


class Resid:
    """Residualiza columnas sobre las covariables con MCO. Se ajusta en entrenamiento y se aplica a prueba."""
    def fit(self, C, M):
        self.lr = LinearRegression().fit(C, M)
        return self

    def __call__(self, C, M):
        return M - self.lr.predict(C)


def fit_model(T, C, y, kind, seed, hyper=None):
    """Ajusta un modelo con tests T (penalizados) y covariables C (sin penalizar).
    hyper=None -> elige hiperparámetros por CV; si no, usa los dados (bootstrap).
    Devuelve una función predict(T, C) y los coeficientes de tests y covariables."""
    rx = Resid().fit(C, T)
    Tr = rx(C, T)
    if kind == "reg":
        ry = Resid().fit(C, y)
        yr = ry(C, y)
        if hyper is None:
            m = fit_reg(Tr, yr, seed)
            hyper = dict(alpha=float(m.alpha_), l1=float(m.l1_ratio_))
        else:
            m = ElasticNet(alpha=hyper["alpha"], l1_ratio=hyper["l1"], max_iter=20000).fit(Tr, yr)
        b = m.coef_
        # Coeficientes de covariables del modelo conjunto (FWL): MCO de (y - T·b) sobre C
        cov_fit = LinearRegression().fit(C, y - T @ b)
        predict = lambda T2, C2: cov_fit.predict(C2) + T2 @ b
        return predict, b, cov_fit.coef_, hyper
    # Logística: tests residualizados + covariables
    X = np.column_stack([Tr, C])
    if hyper is None:
        m = fit_clf(X, y, seed)
        hyper = dict(C=float(m.C_[0]), l1=float(m.l1_ratio_[0]))
    else:
        m = LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=hyper["l1"], C=hyper["C"],
                               max_iter=5000).fit(X, y)
    coef = m.coef_.ravel()
    predict = lambda T2, C2: m.predict_proba(np.column_stack([rx(C2, T2), C2]))[:, 1]
    return predict, coef[:T.shape[1]], coef[T.shape[1]:], hyper


def baseline(C, y, kind):
    """Modelo solo con covariables, sin penalizar."""
    if kind == "reg":
        return LinearRegression().fit(C, y).predict
    m = LogisticRegression(penalty=None, max_iter=5000).fit(C, y)
    return lambda C2: m.predict_proba(C2)[:, 1]


def nested_score(T, C, y, kind):
    """CV anidada: 5 pliegues externos; residualización e hiperparámetros se ajustan dentro de cada pliegue.
    Devuelve (métrica del modelo completo, métrica solo covariables) sobre predicciones fuera de muestra."""
    pred, pbase = np.zeros(len(y)), np.zeros(len(y))
    outer = (StratifiedKFold if kind == "clf" else KFold)(5, shuffle=True, random_state=SEED + 1)
    for tr, te in outer.split(T, y if kind == "clf" else None):
        f, *_ = fit_model(T[tr], C[tr], y[tr], kind, SEED)
        pred[te] = f(T[te], C[te])
        pbase[te] = baseline(C[tr], y[tr], kind)(C[te])
    metric = roc_auc_score if kind == "clf" else r2_score
    return metric(y, pred), metric(y, pbase)


def run(name, s, cols, cv, ycol, kind):
    s = s.dropna(subset=[ycol])
    T, C = s[cols].values.astype(float), s[cv].values.astype(float)
    y = s[ycol].values.astype(float) if kind == "reg" else s[ycol].astype(int).values

    # Ajuste final con todos los casos
    _, bt, bc, hyper = fit_model(T, C, y, kind, SEED)
    coef = np.r_[bt, bc]

    # Rendimiento CV anidada: tests + covariables vs solo covariables
    score, base = nested_score(T, C, y, kind)

    # Bootstrap con hiperparámetros fijos (residualización reajustada en cada remuestra)
    rng = np.random.default_rng(SEED)
    boot = []
    for _ in range(NBOOT):
        i = rng.integers(0, len(y), len(y))
        if kind == "clf" and len(np.unique(y[i])) < 2:
            continue
        _, b1, b2, _ = fit_model(T[i], C[i], y[i], kind, SEED, hyper)
        boot.append(np.r_[b1, b2])
    boot = np.array(boot)
    freq = (np.abs(boot) > 1e-8).mean(axis=0)
    if kind == "reg":
        freq[len(cols):] = np.nan        # covariables sin penalizar: siempre "seleccionadas", no aplica

    names = cols + cv
    R = pd.DataFrame({
        "modelo": name,
        "variable": [n[:-2] for n in names],                    # sin sufijo "_z"
        "tipo": ["test"] * len(cols) + ["covariable"] * len(cv),
        "coef_std": coef,
        "boot_lo": np.percentile(boot, 2.5, axis=0),
        "boot_hi": np.percentile(boot, 97.5, axis=0),
        "freq_seleccion": freq})
    metric = "AUC_cv_anidada" if kind == "clf" else "R2_cv_anidada"
    hp = ", ".join(f"{k}={v:.3g}" for k, v in hyper.items())
    print(f"\n--- {name} | N={len(y)} | {hp}\n    {metric}: tests+covariables={score:.3f} | solo covariables={base:.3f}"
          f" | incremento={score - base:+.3f}")
    print(R[R.tipo == "test"].sort_values("freq_seleccion", ascending=False)
          [["variable", "coef_std", "boot_lo", "boot_hi", "freq_seleccion"]].round(3).to_string(index=False))
    R["N"], R["score_cv"], R["score_solo_covariables"], R["metrica"] = len(y), score, base, metric
    return R


# =====================================================================
# 4. MAIN
# =====================================================================
def main():
    os.makedirs(OUT, exist_ok=True)
    s, tests, cols, cv = prep()

    R = pd.concat([
        run("M1_severidad_con_dolor", s, cols, cv, "pain_sev", "reg"),
        run("M2_CP1_si_no", s, cols, cv, "cp1", "clf"),
        run("M3_amenaza", s, cols, cv, "threat_z", "reg"),
        run("M4_privacion", s, cols, cv, "deprivation_z", "reg")])
    R.to_csv(os.path.join(OUT, "enet_coeficientes.csv"), index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)
    for ax, (n, g) in zip(axes, R[R.tipo == "test"].groupby("modelo", sort=False)):
        y = np.arange(len(g))
        ax.barh(y, g["coef_std"], color=np.where(g["freq_seleccion"] >= SELECTION_THRESHOLD, "#1f5f8b", "#b8c9d6"))
        ax.errorbar(g["coef_std"], y, xerr=[g["coef_std"] - g["boot_lo"], g["boot_hi"] - g["coef_std"]],
                    fmt="none", ecolor="#444", lw=0.8)
        ax.axvline(0, color="grey", lw=1)
        ax.set_yticks(y)
        ax.set_yticklabels(g["variable"])
        ax.invert_yaxis()
        ax.set_title(f"{n} (N={g['N'].iloc[0]})\n{g['metrica'].iloc[0]}={g['score_cv'].iloc[0]:.2f}"
                     f" (solo cov.={g['score_solo_covariables'].iloc[0]:.2f})", fontsize=9)
    fig.suptitle(f"Elastic Net: coeficientes de los tests, ajustados por covariables sin penalizar"
                 f" (oscuro = seleccionado en ≥{SELECTION_THRESHOLD:.0%} de bootstraps)", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "enet_pesos_tests.png"), dpi=150)
    if "agg" not in matplotlib.get_backend().lower():       # mostrar en pantalla si hay backend interactivo
        plt.show()
    plt.close()
    print(f"\nArchivos en: {os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()