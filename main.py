"""
GARCH NASDAQ ANOMALY DETECTOR
==============================
Detección de anomalías de volatilidad sobre QQQ usando GARCH(1,1)
con Z-score rodante y clasificación automática por severidad.
"""

import os
import warnings
import sys
from pathlib import Path

# Forzar UTF-8 en la consola de Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from arch import arch_model
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

# ── Rutas ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
OUT_DIR    = BASE_DIR / "outputs"
DATA_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

# ── Constantes ─────────────────────────────────────────────────────────────────
TICKER      = "QQQ"
START_DATE  = "2015-01-01"
ROLL_WINDOW = 60
ANNUAL_K    = np.sqrt(252)

REGIMES = {
    "Normal":           ("#2ecc71", lambda z: np.abs(z) < 1.5),
    "Vol Elevada":      ("#f1c40f", lambda z: (z >= 1.5) & (z < 2.5)),
    "Anomalía":         ("#e67e22", lambda z: (z >= 2.5) & (z < 3.5)),
    "Anomalía Extrema": ("#e74c3c", lambda z: z >= 3.5),
    "Vol Comprimida":   ("#3498db", lambda z: z < -1.5),
}


# ══════════════════════════════════════════════════════════════════════════════
# PASO 1 — DATOS
# ══════════════════════════════════════════════════════════════════════════════

def download_and_prepare_data(
    ticker: str = TICKER,
    start: str = START_DATE,
) -> pd.DataFrame:
    """
    Descarga precios ajustados y calcula log-retornos diarios.

    Parameters
    ----------
    ticker : str
        Símbolo del activo (default: 'QQQ').
    start : str
        Fecha de inicio en formato 'YYYY-MM-DD'.

    Returns
    -------
    pd.DataFrame
        DataFrame con columnas ['price', 'log_return'], índice DatetimeIndex.
    """
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        raw = t.history(start=start, auto_adjust=True)
        if raw.empty:
            raise ValueError(f"No se obtuvieron datos para {ticker}.")
        prices = raw["Close"].squeeze().dropna()
        prices.index = prices.index.tz_localize(None)  # eliminar timezone
    except Exception as exc:
        print(f"\n[ERROR] Fallo al descargar datos de {ticker}: {exc}")
        print("Verifica tu conexión a internet e intenta de nuevo.")
        sys.exit(1)

    log_ret = np.log(prices / prices.shift(1)).dropna()

    df = pd.DataFrame({"price": prices, "log_return": log_ret}).dropna()

    # Test ADF para verificar estacionariedad de retornos
    adf_stat, adf_p, *_ = adfuller(df["log_return"], autolag="AIC")
    stationary = "SÍ" if adf_p < 0.05 else "NO"
    print(f"  ADF test log-retornos -> estadistico={adf_stat:.4f}, p={adf_p:.4f} -> Estacionaria: {stationary}")

    out_path = DATA_DIR / "qqq_returns.csv"
    df.to_csv(out_path)
    print(f"  Datos guardados en {out_path}  ({len(df)} observaciones)")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# PASO 2 — MODELO GARCH(1,1)
# ══════════════════════════════════════════════════════════════════════════════

def fit_garch(
    returns: pd.Series,
) -> Tuple[object, pd.Series]:
    """
    Ajusta un modelo GARCH(1,1) con distribución t de Student.

    Parameters
    ----------
    returns : pd.Series
        Serie de log-retornos diarios.

    Returns
    -------
    Tuple[ARCHModelResult, pd.Series]
        (resultado del modelo, volatilidad condicional anualizada)
    """
    # Escalar retornos a % para estabilidad numérica del optimizador
    ret_pct = returns * 100

    am = arch_model(
        ret_pct,
        mean="Constant",
        vol="GARCH",
        p=1,
        q=1,
        dist="t",
    )
    res = am.fit(disp="off", show_warning=False)

    # Volatilidad condicional anualizada (en %)
    cond_vol = res.conditional_volatility * ANNUAL_K
    cond_vol.name = "cond_vol_annual"

    return res, cond_vol


# ══════════════════════════════════════════════════════════════════════════════
# PASO 3 — Z-SCORE RODANTE
# ══════════════════════════════════════════════════════════════════════════════

def compute_rolling_zscore(
    cond_vol: pd.Series,
    window: int = ROLL_WINDOW,
) -> Tuple[pd.Series, pd.Series]:
    """
    Calcula el Z-score rodante de la volatilidad condicional y asigna régimen.

    Parameters
    ----------
    cond_vol : pd.Series
        Volatilidad condicional anualizada.
    window : int
        Ventana rodante en días (default: 60).

    Returns
    -------
    Tuple[pd.Series, pd.Series]
        (z_score, regime_label)
    """
    roll_mean = cond_vol.rolling(window).mean()
    roll_std  = cond_vol.rolling(window).std()
    z_score   = (cond_vol - roll_mean) / roll_std
    z_score.name = "z_score"

    regime = pd.Series("Normal", index=z_score.index, name="regime")
    # Orden importa: más específico primero
    regime[z_score < -1.5]                          = "Vol Comprimida"
    regime[(z_score >= 1.5) & (z_score < 2.5)]     = "Vol Elevada"
    regime[(z_score >= 2.5) & (z_score < 3.5)]     = "Anomalía"
    regime[z_score >= 3.5]                          = "Anomalía Extrema"

    return z_score, regime


# ══════════════════════════════════════════════════════════════════════════════
# PASO 4 — TABLA DE EVENTOS
# ══════════════════════════════════════════════════════════════════════════════

def build_anomaly_table(
    df: pd.DataFrame,
    cond_vol: pd.Series,
    z_score: pd.Series,
    regime: pd.Series,
    threshold: float = 2.5,
) -> pd.DataFrame:
    """
    Construye y guarda la tabla de eventos anómalos.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con precio y log-retorno.
    cond_vol : pd.Series
        Volatilidad condicional anualizada.
    z_score : pd.Series
        Z-score rodante.
    regime : pd.Series
        Etiqueta de régimen.
    threshold : float
        Umbral mínimo de |Z| para considerar anomalía (default: 2.5).

    Returns
    -------
    pd.DataFrame
        Tabla de eventos ordenada por z_score descendente.
    """
    combined = pd.DataFrame({
        "precio_qqq":      df["price"],
        "retorno_dia":     df["log_return"],
        "vol_condicional": cond_vol,
        "z_score":         z_score,
        "clasificacion":   regime,
    }).dropna()

    anomalies = combined[np.abs(combined["z_score"]) >= threshold].copy()
    anomalies = anomalies.sort_values("z_score", ascending=False)

    out_path = OUT_DIR / "anomaly_events.csv"
    anomalies.to_csv(out_path)
    print(f"  Tabla de anomalías guardada en {out_path}  ({len(anomalies)} filas)")

    print("\n  TOP 15 EVENTOS HISTÓRICOS:")
    print(f"  {'Fecha':<12} {'Precio':>8} {'Retorno':>9} {'Vol%':>8} {'Z-score':>8}  Clasificación")
    print("  " + "─" * 70)
    for date, row in anomalies.head(15).iterrows():
        d = str(date)[:10]
        print(
            f"  {d:<12} {row.precio_qqq:>8.2f} {row.retorno_dia*100:>8.2f}%"
            f" {row.vol_condicional:>7.1f}% {row.z_score:>8.2f}  {row.clasificacion}"
        )

    return anomalies, combined


# ══════════════════════════════════════════════════════════════════════════════
# PASO 5 — VISUALIZACIONES
# ══════════════════════════════════════════════════════════════════════════════

REGIME_COLORS = {
    "Normal":           "#2ecc7133",
    "Vol Elevada":      "#f1c40f55",
    "Anomalía":         "#e67e2277",
    "Anomalía Extrema": "#e74c3caa",
    "Vol Comprimida":   "#3498db44",
}


def _shade_regimes(ax: plt.Axes, dates: pd.DatetimeIndex, regime: pd.Series) -> None:
    """Sombrea el fondo del eje según el régimen de cada período."""
    prev_regime = None
    start_date  = None
    for date, reg in regime.items():
        if reg != prev_regime:
            if prev_regime is not None:
                ax.axvspan(start_date, date, color=REGIME_COLORS.get(prev_regime, "#ffffff00"), lw=0)
            prev_regime = reg
            start_date  = date
    if prev_regime is not None:
        ax.axvspan(start_date, dates[-1], color=REGIME_COLORS.get(prev_regime, "#ffffff00"), lw=0)


def plot_panel_principal(
    combined: pd.DataFrame,
    cond_vol: pd.Series,
    z_score: pd.Series,
    regime: pd.Series,
) -> None:
    """
    Genera el panel principal con 3 subplots: precio, volatilidad y Z-score.

    Parameters
    ----------
    combined : pd.DataFrame
        DataFrame con todas las series alineadas.
    cond_vol : pd.Series
        Volatilidad condicional anualizada.
    z_score : pd.Series
        Z-score rodante.
    regime : pd.Series
        Etiqueta de régimen.
    """
    fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True)
    fig.suptitle("GARCH Anomaly Detector — QQQ", fontsize=16, fontweight="bold", y=0.98)
    dates = combined.index

    # ── Subplot 1: Precio ──────────────────────────────────────────────────────
    ax1 = axes[0]
    _shade_regimes(ax1, dates, regime.reindex(combined.index))
    ax1.plot(combined.index, combined["precio_qqq"], color="#2c3e50", lw=1.2, label="QQQ Close")
    ax1.set_ylabel("Precio (USD)", fontsize=10)
    ax1.set_title("Precio QQQ con regímenes de volatilidad", fontsize=11)
    # leyenda de regímenes
    legend_patches = [
        mpatches.Patch(color="#2ecc71", alpha=0.5, label="Normal"),
        mpatches.Patch(color="#f1c40f", alpha=0.5, label="Vol Elevada"),
        mpatches.Patch(color="#e67e22", alpha=0.6, label="Anomalía"),
        mpatches.Patch(color="#e74c3c", alpha=0.7, label="Anomalía Extrema"),
        mpatches.Patch(color="#3498db", alpha=0.4, label="Vol Comprimida"),
    ]
    ax1.legend(handles=legend_patches, loc="upper left", fontsize=8, ncol=3)

    # ── Subplot 2: Volatilidad condicional ─────────────────────────────────────
    ax2 = axes[1]
    vol_aligned = cond_vol.reindex(combined.index)
    ax2.plot(combined.index, vol_aligned, color="#2980b9", lw=1.2, label="Vol GARCH anualizada")
    p75  = vol_aligned.quantile(0.75)
    p90  = vol_aligned.quantile(0.90)
    p95  = vol_aligned.quantile(0.95)
    for pct, lbl, clr in [(p75, "P75", "#f39c12"), (p90, "P90", "#e67e22"), (p95, "P95", "#e74c3c")]:
        ax2.axhline(pct, ls="--", lw=0.9, color=clr, alpha=0.8, label=f"{lbl}: {pct:.1f}%")
    ax2.set_ylabel("Volatilidad anualizada (%)", fontsize=10)
    ax2.set_title("Volatilidad condicional GARCH(1,1)", fontsize=11)
    ax2.legend(loc="upper right", fontsize=8)

    # ── Subplot 3: Z-score ─────────────────────────────────────────────────────
    ax3 = axes[2]
    z_aligned = z_score.reindex(combined.index)
    ax3.plot(combined.index, z_aligned, color="#8e44ad", lw=0.9, alpha=0.85, label="Z-score (60d)")
    ax3.fill_between(combined.index, z_aligned, 0,
                     where=z_aligned >= 2.5, color="#e74c3c", alpha=0.3, label="|Z|≥2.5")
    ax3.fill_between(combined.index, z_aligned, 0,
                     where=z_aligned <= -1.5, color="#3498db", alpha=0.3, label="Z<-1.5")
    for level, clr, ls in [(1.5, "#f1c40f", "--"), (2.5, "#e67e22", "--"), (3.5, "#e74c3c", "-"),
                            (-1.5, "#3498db", "--")]:
        ax3.axhline(level, color=clr, ls=ls, lw=0.9, alpha=0.8)
    ax3.set_ylabel("Z-score", fontsize=10)
    ax3.set_xlabel("Fecha", fontsize=10)
    ax3.set_title("Z-score rodante (ventana 60 días)", fontsize=11)
    ax3.legend(loc="upper right", fontsize=8)

    for ax in axes:
        ax.grid(True, alpha=0.25, lw=0.5)

    plt.tight_layout()
    out_path = OUT_DIR / "panel_principal.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Gráfico guardado: {out_path}")


def plot_distribucion_vol(cond_vol: pd.Series) -> None:
    """
    Histograma de la volatilidad condicional con overlay normal y percentiles.

    Parameters
    ----------
    cond_vol : pd.Series
        Volatilidad condicional anualizada.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    data = cond_vol.dropna()

    ax.hist(data, bins=60, density=True, color="#2980b9", alpha=0.6, label="Distribución empírica")

    mu, sigma = data.mean(), data.std()
    x = np.linspace(data.min(), data.max(), 300)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), "r-", lw=2, label=f"Normal teórica (μ={mu:.1f}%, σ={sigma:.1f}%)")

    for pct, clr in [(75, "#f39c12"), (90, "#e67e22"), (95, "#e74c3c"), (99, "#8e44ad")]:
        val = np.percentile(data, pct)
        ax.axvline(val, color=clr, ls="--", lw=1.4, label=f"P{pct}: {val:.1f}%")

    ax.set_title("Distribución de la Volatilidad Condicional GARCH — QQQ", fontsize=13, fontweight="bold")
    ax.set_xlabel("Volatilidad anualizada (%)", fontsize=11)
    ax.set_ylabel("Densidad", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = OUT_DIR / "distribucion_vol.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Gráfico guardado: {out_path}")


def plot_anomalias_zoom(
    combined: pd.DataFrame,
    anomalies: pd.DataFrame,
    cond_vol: pd.Series,
    context_days: int = 30,
) -> None:
    """
    Zoom en los 5 períodos de anomalía extrema más importantes.

    Parameters
    ----------
    combined : pd.DataFrame
        DataFrame con precio y series de volatilidad.
    anomalies : pd.DataFrame
        Tabla de eventos anómalos ordenada por z_score.
    cond_vol : pd.Series
        Volatilidad condicional anualizada.
    context_days : int
        Días de contexto a cada lado del evento (default: 30).
    """
    extreme = anomalies[anomalies["clasificacion"] == "Anomalía Extrema"]
    if extreme.empty:
        extreme = anomalies.head(5)
    top5 = extreme.head(5)

    n = len(top5)
    if n == 0:
        print("  No hay suficientes anomalías extremas para el gráfico de zoom.")
        return

    fig, axes = plt.subplots(n, 2, figsize=(16, 4 * n))
    if n == 1:
        axes = np.array([axes])
    fig.suptitle("Zoom — Top 5 Períodos de Anomalía Extrema", fontsize=14, fontweight="bold")

    all_dates = combined.index
    vol_aligned = cond_vol.reindex(combined.index)

    for i, (event_date, row) in enumerate(top5.iterrows()):
        idx  = all_dates.get_loc(event_date) if event_date in all_dates else None
        if idx is None:
            continue
        lo   = max(0, idx - context_days)
        hi   = min(len(all_dates), idx + context_days + 1)
        window_dates = all_dates[lo:hi]

        # Panel izquierdo: Precio
        ax_p = axes[i, 0]
        ax_p.plot(window_dates, combined.loc[window_dates, "precio_qqq"], color="#2c3e50", lw=1.3)
        ax_p.axvline(event_date, color="#e74c3c", ls="--", lw=1.5, alpha=0.9)
        ax_p.set_title(f"{str(event_date)[:10]}  |  Z={row.z_score:.2f}  |  {row.clasificacion}", fontsize=10)
        ax_p.set_ylabel("Precio (USD)", fontsize=9)
        ax_p.grid(True, alpha=0.25)

        # Panel derecho: Volatilidad
        ax_v = axes[i, 1]
        ax_v.plot(window_dates, vol_aligned.loc[window_dates], color="#e74c3c", lw=1.3)
        ax_v.axvline(event_date, color="#c0392b", ls="--", lw=1.5, alpha=0.9)
        ax_v.set_ylabel("Vol anualizada (%)", fontsize=9)
        ax_v.grid(True, alpha=0.25)

    plt.tight_layout()
    out_path = OUT_DIR / "anomalias_zoom.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Gráfico guardado: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# PASO 6 — REPORTE DE CONSOLA
# ══════════════════════════════════════════════════════════════════════════════

def print_report(
    df: pd.DataFrame,
    garch_result: object,
    cond_vol: pd.Series,
    z_score: pd.Series,
    anomalies: pd.DataFrame,
) -> None:
    """
    Imprime el reporte completo en consola.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de precios y retornos.
    garch_result : ARCHModelResult
        Resultado del modelo GARCH ajustado.
    cond_vol : pd.Series
        Volatilidad condicional anualizada.
    z_score : pd.Series
        Z-score rodante.
    anomalies : pd.DataFrame
        Tabla de eventos anómalos.
    """
    params = garch_result.params
    pvals  = garch_result.pvalues
    vol    = cond_vol.dropna()
    z      = z_score.dropna()

    vol_min_date = vol.idxmin()
    vol_max_date = vol.idxmax()

    total_days     = len(z)
    anomaly_days   = int((np.abs(z) >= 2.5).sum())
    extreme_events = int((z >= 3.5).sum())
    last_event_idx = anomalies.index[0] if not anomalies.empty else None

    omega  = params.get("omega",  params.get("Const",  float("nan")))
    alpha1 = params.get("alpha[1]", float("nan"))
    beta1  = params.get("beta[1]",  float("nan"))
    persist = alpha1 + beta1

    p_omega  = pvals.get("omega",    pvals.get("Const",  float("nan")))
    p_alpha  = pvals.get("alpha[1]", float("nan"))
    p_beta   = pvals.get("beta[1]",  float("nan"))

    SEP = "═" * 51
    print(f"\n{SEP}")
    print("  GARCH NASDAQ ANOMALY DETECTOR — QQQ")
    print(SEP)
    print(f"  Período analizado    : {str(df.index[0])[:10]} → {str(df.index[-1])[:10]}")
    print(f"  Observaciones        : {len(df)} días")
    print()
    print("  PARÁMETROS GARCH(1,1):")
    print(f"    omega  : {omega:.6f}  (p={p_omega:.4f})")
    print(f"    alpha1 : {alpha1:.6f}  (p={p_alpha:.4f})")
    print(f"    beta1  : {beta1:.6f}  (p={p_beta:.4f})")
    print(f"    Persistencia (α+β): {persist:.4f}")
    print()
    print("  ESTADÍSTICAS DE VOLATILIDAD:")
    print(f"    Vol media anualizada : {vol.mean():.1f}%")
    print(f"    Vol mínima           : {vol.min():.1f}%  ({str(vol_min_date)[:10]})")
    print(f"    Vol máxima           : {vol.max():.1f}%  ({str(vol_max_date)[:10]})")
    print(f"    Percentil 90         : {vol.quantile(0.90):.1f}%")
    print()
    print("  ANOMALÍAS DETECTADAS:")
    print(f"    Total días anómalos  : {anomaly_days} ({anomaly_days/total_days*100:.1f}% del período)")
    print(f"    Anomalías extremas   : {extreme_events} eventos")
    if last_event_idx is not None:
        print(f"    Último evento        : {str(anomalies.index[0])[:10]}  (Z={anomalies.iloc[0].z_score:.2f})")
    print()
    print("  TOP 5 ANOMALÍAS HISTÓRICAS:")
    for rank, (date, row) in enumerate(anomalies.head(5).iterrows(), 1):
        print(
            f"    {rank}. {str(date)[:10]}  |  Vol={row.vol_condicional:.1f}%"
            f"  |  Z={row.z_score:+.2f}  |  [{row.clasificacion}]"
        )
    print(SEP)
    print(f"\n  Proyecto en: {BASE_DIR.resolve()}\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Ejecuta el pipeline completo de detección de anomalías GARCH."""

    print("\n[1/6] Descargando y preparando datos...")
    df = download_and_prepare_data()

    print("\n[2/6] Ajustando GARCH(1,1)...")
    garch_result, cond_vol = fit_garch(df["log_return"])
    print(f"  Log-likelihood: {garch_result.loglikelihood:.2f} | AIC: {garch_result.aic:.2f}")

    print("\n[3/6] Calculando Z-score rodante...")
    z_score, regime = compute_rolling_zscore(cond_vol)
    print(f"  Z-score range: [{z_score.min():.2f}, {z_score.max():.2f}]")

    print("\n[4/6] Construyendo tabla de anomalías...")
    anomalies, combined = build_anomaly_table(df, cond_vol, z_score, regime)

    print("\n[5/6] Generando visualizaciones...")
    plot_panel_principal(combined, cond_vol, z_score, regime)
    plot_distribucion_vol(cond_vol)
    plot_anomalias_zoom(combined, anomalies, cond_vol)

    print("\n[6/6] Reporte final:")
    print_report(df, garch_result, cond_vol, z_score, anomalies)

    # Verificación final
    checks = [
        (OUT_DIR / "panel_principal.png").exists(),
        (OUT_DIR / "distribucion_vol.png").exists(),
        (OUT_DIR / "anomalias_zoom.png").exists(),
        (OUT_DIR / "anomaly_events.csv").exists(),
    ]
    names = ["panel_principal.png", "distribucion_vol.png", "anomalias_zoom.png", "anomaly_events.csv"]
    print("  VERIFICACIÓN DE OUTPUTS:")
    for name, ok in zip(names, checks):
        status = "✓" if ok else "✗"
        print(f"    {status}  outputs/{name}")
    if not all(checks):
        print("\n  [ADVERTENCIA] Algunos archivos no se generaron correctamente.")


if __name__ == "__main__":
    main()
