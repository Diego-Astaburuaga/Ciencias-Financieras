# Caso 1 – Analyzing the Impact of COVID-19 on Pension Fund Switching in Chile

**Curso:** Econometría Avanzada  
**Tipo de trabajo:** Caso #1 – Modelamiento Binario  
**Autores:** Werner Kristjanpoller, Camilo Aguilar, Diego Astaburuaga, Felipe Mora, Rodrigo Serrano  
**Institución:** Universidad Técnica Federico Santa María (UTFSM)  
**Fecha:** Octubre 2024

---

## 🎯 Objetivo

Analizar el comportamiento de los afiliados al sistema de pensiones chileno durante la pandemia de COVID-19 (2020–2022), estudiando los factores que influyeron en la decisión de cambio de fondo previsional. Se busca determinar si la crisis sanitaria generó comportamientos distintos a los de un período normal (2018–2019), y cuantificar el efecto de variables demográficas y de educación financiera sobre dichas decisiones.

---

## 📐 Métodos Utilizados

- **Modelo Probit** y **Modelo Logit**: modelos de elección binaria estimados por máxima verosimilitud para determinar la probabilidad de que un afiliado realice un cambio de fondo.
- **Efectos marginales**: para interpretar el impacto cuantitativo de cada variable explicativa.
- **Análisis descriptivo**: comparación estadística entre el período pre-pandemia (2018–2019) y el período pandémico (2020–2022).
- **Variables explicativas**: edad, sexo, nivel de ingresos, conocimiento financiero, tolerancia al riesgo y fondo actual del afiliado.

---

## 📊 Datos

Los datos fueron obtenidos desde:
- **Superintendencia de Pensiones** (`spensiones.cl`): base de cambios de fondo (`base_cambio_fondos.csv`) y características de afiliados (`caracteristicas_afiliados.csv`).
- **Banco Central de Chile** (`bcentral.cl`): para el valor de la Unidad de Fomento (UF).
- **Período de análisis:** 2020-01-01 a 2022-12-31 (con grupo de control 2018–2019).

---

## 🔍 Lo que se hizo

1. Se revisó la literatura sobre planes de retiro de contribución definida (DC) y el sistema de pensiones chileno (AFP).
2. Se construyó una base de datos combinando información de cambios de fondo con características sociodemográficas y financieras de los afiliados.
3. Se estimaron modelos Probit y Logit para identificar los determinantes de los cambios de fondo durante la pandemia.
4. Se compararon los resultados del período COVID con un período de control pre-pandemia.
5. Se analizaron los efectos marginales de cada variable para cuantificar su impacto en la probabilidad de cambio.

---

## 📈 Resultados

- El período COVID-19 tuvo **implicancias significativas** en el comportamiento de los afiliados respecto a sus fondos de pensiones.
- Se encontraron **relaciones entre las características individuales** (edad, ingreso, conocimiento financiero) y el número de cambios de fondo realizados.
- Los individuos con **menor conocimiento financiero** tendieron a elegir fondos por defecto, mientras que los de mayor conocimiento financiero y mayor tolerancia al riesgo optaron por fondos más riesgosos.
- La **tolerancia al riesgo disminuye con la edad** y aumenta con el nivel de ingresos y educación financiera.
- El estudio concluye que comprender estos comportamientos es crucial para la gestión de fondos en situaciones no convencionales.

---

## 📄 Archivo

- [`Estudio_Fondo_Pensiones_Chile_En_Covid.pdf`](./Estudio_Fondo_Pensiones_Chile_En_Covid.pdf) — Presentación completa del estudio.
