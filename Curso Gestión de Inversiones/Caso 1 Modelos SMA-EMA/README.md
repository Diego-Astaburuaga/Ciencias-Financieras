# Caso 1 – Modelos SMA y EMA aplicados a una Acción Tecnológica del NASDAQ

**Curso:** Gestión de Inversiones (ICN256)  
**Autores:** Diego Astaburuaga, Sebastián Flández  
**Institución:** Universidad Técnica Federico Santa María (UTFSM)  
**Acción analizada:** MicroStrategy Incorporated (MSTR)

---

## 🎯 Objetivo

Implementar y evaluar estrategias de trading basadas en cruces de medias móviles simples (SMA) y exponenciales (EMA) sobre la acción de MicroStrategy (MSTR), una empresa tecnológica listada en el NASDAQ con fuerte exposición a Bitcoin. Se busca comparar el desempeño de distintas combinaciones de períodos cortos y largos contra una estrategia pasiva de *buy & hold*.

---

## 📐 Métodos Utilizados

- **Simple Moving Average (SMA)**: promedio aritmético de los precios de cierre en una ventana de N períodos.
- **Exponential Moving Average (EMA)**: media móvil ponderada exponencialmente, que da mayor peso a los datos recientes mediante el coeficiente de suavizado α = 2/(N+1).
- **Estrategia de cruce de medias (*crossover*)**: señal de compra cuando la media móvil rápida cruza hacia arriba a la lenta, y señal de venta en el cruce inverso.
- **Backtesting**: evaluación del retorno acumulado de cada estrategia sobre datos históricos de 5 años.
- **Combinaciones evaluadas:** (12, 26), (10, 50), (20, 200), (50, 200) para ambos tipos de media móvil.

---

## 📊 Datos

- **Activo:** Acción de MicroStrategy (MSTR), NASDAQ.
- **Período:** 5 años de datos históricos diarios (descargados vía `yfinance`).
- **Variables:** precios de cierre ajustados.

---

## 🔍 Lo que se hizo

1. Se definieron matemáticamente los modelos SMA y EMA y los períodos clásicos de trading.
2. Se implementó una función de backtesting en Python que descarga datos, calcula las medias móviles, genera señales de trading y calcula los retornos acumulados, con opción de incluir costos de transacción.
3. Se evaluaron 8 estrategias en total (4 combinaciones de períodos × 2 tipos de media móvil) y se compararon contra la estrategia pasiva *buy & hold*.
4. Se construyó una tabla comparativa con las diferencias de retorno acumulado respecto a la estrategia pasiva para cada estrategia.

---

## 📈 Resultados

- Se obtuvo una tabla comparativa de retornos acumulados para todas las estrategias evaluadas, ordenada de mayor a menor diferencia respecto al *buy & hold*.
- Los resultados permiten identificar qué combinaciones de períodos y tipo de media móvil generan mejor rendimiento en el horizonte analizado para este activo volátil.
- Las estrategias de medias móviles con períodos más cortos tienden a reaccionar más rápido a los movimientos del precio, pero también generan más señales falsas en mercados muy volátiles como MSTR.

---

## 📄 Archivos

- [`Modelos_SMA-EMA.ipynb`](./Modelos_SMA-EMA.ipynb) — Notebook con la implementación completa en Python.
- [`Modelos_SMA-EMA.pdf`](./Modelos_SMA-EMA.pdf) — Versión en PDF del informe.
