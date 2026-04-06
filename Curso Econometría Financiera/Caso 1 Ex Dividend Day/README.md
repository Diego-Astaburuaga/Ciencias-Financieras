# Caso 1 – Ex-Dividend Day: Caso Chileno Actualizado

**Curso:** Econometría Financiera (PII431)  
**Autores:** Diego Astaburuaga, Sebastián Flández  
**Institución:** Universidad Técnica Federico Santa María (UTFSM)  
**Profesor:** Werner Kristjanpoller

---

## 🎯 Objetivo

Investigar el efecto ex-dividend en el mercado accionario chileno, analizando si los precios de las acciones se ajustan de forma completa o parcial al valor del dividendo en la fecha ex-dividend. Se busca actualizar la evidencia empírica chilena incorporando datos recientes (2017–2025), y evaluar la relación entre el *Price Drop Ratio* (PDR) y el *dividend yield*, controlando por condiciones de mercado y retornos esperados ajustados por riesgo.

---

## 📐 Métodos Utilizados

- **Estudio de eventos (*event study*)**: para identificar la reacción del precio de la acción en torno a la fecha ex-dividend.
- **Price Drop Ratio (PDR)**: métrica que mide la caída del precio en relación al dividendo pagado.
- **Adjusted Market Ratio (AMR)**: controla los movimientos del mercado usando el índice IPSA.
- **CAPM**: estimación de retornos esperados usando betas individuales calculados por regresión histórica.
- **Retornos Anormales (AR) y Exceso de Retorno**: medición de desviaciones respecto al retorno esperado.
- **OLS (Mínimos Cuadrados Ordinarios)** y **RLM (Regresión Lineal Robusta)**: para evaluar la relación entre PDR/AR y variables explicativas como *dividend yield*, volumen, volatilidad y momentum.
- **Correlaciones de Spearman y Kendall**: para evaluar robustez de resultados no paramétricos.

---

## 📊 Datos

- **Universo:** 30 acciones listadas en la Bolsa de Santiago, de diversas industrias.
- **Período:** 11 de abril de 2017 al 31 de marzo de 2025 (8 años de datos).
- **Variables:** precios de cierre diarios, volumen transado, valor del IPSA, dividendos pagados.
- **Fuente:** datos descargados desde la Bolsa de Santiago y fuentes financieras públicas.

---

## 🔍 Lo que se hizo

1. Se recopiló información de precios y dividendos para 30 acciones chilenas durante 8 años.
2. Se calculó el PDR para cada evento de dividendo y se estimaron retornos anormales usando CAPM con betas individuales.
3. Se controló el efecto de mercado mediante el AMR basado en el IPSA.
4. Se ajustaron modelos OLS y RLM para evaluar si el *dividend yield*, volumen, volatilidad y momentum explican las variaciones en PDR y AR.
5. Se aplicaron métricas de correlación no paramétricas (Spearman y Kendall) para verificar la consistencia de los resultados.
6. Se filtraron observaciones atípicas para evaluar la robustez de los modelos.

---

## 📈 Resultados

- Los valores de PDR muestran **alta variabilidad** entre empresas y a lo largo del tiempo, lo que evidencia que el mercado chileno no ajusta de forma uniforme los precios al valor del dividendo.
- La **correlación entre PDR y dividend yield es débil** en la mayoría de las especificaciones, consistente con la evidencia previa de Castillo & Jakob (2006), quienes encontraron un PDR promedio de 0.76 para el período 1989–2004.
- Los modelos de regresión robusta identificaron un rol estadísticamente significativo del **momentum de corto plazo** y la **volatilidad** en la explicación del PDR y los retornos anormales.
- El **exceso de retorno y el retorno anormal (AR) están altamente correlacionados**, confirmando su vínculo teórico.
- Los resultados sugieren que el mercado chileno no es completamente eficiente en la incorporación del dividendo al precio, con fricciones relacionadas a la regulación local, el perfil heterogéneo de inversionistas y las condiciones de mercado.

---

## 📄 Archivo

- [`Ex_dividend_day.pdf`](./Ex_dividend_day.pdf) — Artículo completo del estudio.
