# GARCH NASDAQ Anomaly Detector

Detección automática de anomalías de volatilidad sobre QQQ usando GARCH(1,1) con Z-score rodante.

## Instalación

```bash
pip install -r requirements.txt
```

## Ejecución

```bash
python main.py
```

## Outputs generados

| Archivo | Descripción |
|---------|-------------|
| `outputs/panel_principal.png` | Panel 3-en-1: precio, volatilidad GARCH y Z-score |
| `outputs/distribucion_vol.png` | Distribución de la volatilidad condicional |
| `outputs/anomalias_zoom.png` | Zoom en los 5 eventos extremos más importantes |
| `outputs/anomaly_events.csv` | Tabla completa de todos los días anómalos |
| `data/qqq_returns.csv` | Log-retornos históricos de QQQ |

## Clasificación de regímenes

| Régimen | Condición Z-score |
|---------|-------------------|
| Normal | \|Z\| < 1.5 |
| Vol Elevada | 1.5 ≤ Z < 2.5 |
| Anomalía | 2.5 ≤ Z < 3.5 |
| Anomalía Extrema | \|Z\| ≥ 3.5 |
| Vol Comprimida | Z < -1.5 |
