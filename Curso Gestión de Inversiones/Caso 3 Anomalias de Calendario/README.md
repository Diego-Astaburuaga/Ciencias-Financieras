# Caso 3 – Análisis de Anomalías de Calendario en Criptomonedas

**Curso:** Gestión de Inversiones (ICN256)  
**Autores:** Diego Astaburuaga, Sebastián Flández, José De los Santos M., Enrique Guerrero González  
**Institución:** Universidad Técnica Federico Santa María (UTFSM)  
**Profesor:** Werner Kristjanpoller  
**Ayudantes:** Daniel Loaiza, Felipe Mora, Valeria Carvajal

---

## 🎯 Objetivo

Identificar y caracterizar la existencia de anomalías de calendario en el mercado de criptomonedas, enfocándose en el efecto día de la semana (*Day-of-the-Week*, DoW) y el efecto de cambio de mes (*Turn-of-the-Month*). Se busca evaluar si estos patrones temporales representan ineficiencias explotables que contradicen la Hipótesis de Mercado Eficiente (EMH) en su forma débil, y construir estrategias de portafolio basadas en ellos.

---

## 📐 Métodos Utilizados

- **Modelos de regresión lineal con variables dummy**: para detectar retornos promedio significativamente distintos por día de la semana y por período del mes.
- **Retornos logarítmicos diarios**: como variable dependiente en los modelos de anomalías.
- **Optimización de portafolio (teoría de Markowitz)**:
  - Portafolio de **máxima rentabilidad**
  - Portafolio de **mínimo riesgo**
  - Portafolio de **máximo Sharpe**
- **Análisis de eficiencia de mercado**: evaluación de si los patrones detectados son estadísticamente robustos.
- **Análisis descriptivo del diseño de criptomonedas**: revisión de mecanismos de consenso, límites de emisión y tipos de blockchain.

---

## 📊 Datos

- **Criptomonedas analizadas:** Bitcoin (BTC), Ethereum (ETH), Solana (SOL), XRP y Cardano (ADA).
- **Período:** 16 de abril de 2024 al 15 de abril de 2025 (1 año de datos diarios).
- **Frecuencia:** datos diarios (el mercado cripto opera 7 días a la semana, 24 horas).
- **Fuente:** plataformas de datos financieros públicos.

---

## 🔍 Lo que se hizo

1. Se recopilaron datos de retornos diarios para las cinco criptomonedas seleccionadas durante un año.
2. Se estimaron modelos de regresión con variables dummy para cada día de la semana y para el efecto de cambio de mes, identificando patrones estadísticamente significativos.
3. Se analizó el efecto enero en el contexto cripto.
4. Se construyeron portafolios óptimos de Markowitz (máxima rentabilidad, mínimo riesgo y máximo Sharpe) para cada día de la semana, evaluando si la estrategia basada en anomalías puede mejorar el rendimiento.
5. Se contextualizó el análisis con eventos geopolíticos recientes (reserva de Bitcoin en EE.UU., escándalo Cryptogate en Argentina).
6. Se describieron aspectos técnicos del diseño de criptomonedas y sus implicancias para la dinámica de mercado.

---

## 📈 Resultados

- Los modelos de regresión muestran que los **p-valores para el efecto día de la semana no son estadísticamente significativos** de forma generalizada, lo que sugiere evidencia limitada de anomalías robustas.
- Los resultados del efecto enero tampoco son concluyentes, con ninguna criptomoneda mostrando coeficientes significativos al nivel estándar.
- La optimización de portafolio revela que **XRP domina en las estrategias de máxima rentabilidad** en la mayoría de los días, mientras que **ETH tiende a dominar en los portafolios de mínimo riesgo**.
- El portafolio de **máximo Sharpe** presenta resultados mixtos por día, con BTC siendo relevante los lunes y SOL los sábados.
- Los retornos de la estrategia basada en anomalías pueden **mejorar levemente**, en particular evitando días de menor rendimiento, pero la evidencia no es suficientemente sólida para rechazar la eficiencia de mercado en su forma débil.

---

## 📄 Archivo

- [`Anomalias de Calendario.pdf`](./Anomalias%20de%20Calendario.pdf) — Artículo completo del estudio.
