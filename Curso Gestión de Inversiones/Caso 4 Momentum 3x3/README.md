# Caso 4 – Estrategia Momentum 3x3 en el Mercado Accionario Colombiano

**Curso:** Gestión de Inversiones (ICN256)  
**Autores:** Diego Astaburuaga, Sebastián Flández, José De los Santos M., Enrique Guerrero González  
**Institución:** Universidad Técnica Federico Santa María (UTFSM)  
**Profesor:** Werner Kristjanpoller  
**Ayudantes:** Daniel Loaiza, Felipe Mora, Valeria Carvajal

---

## 🎯 Objetivo

Evaluar la efectividad de una estrategia cuantitativa momentum 3x3 en el mercado accionario colombiano, utilizando datos de 15 acciones listadas en la Bolsa de Valores de Colombia (BVC) durante el período mayo 2020 – mayo 2025. Se busca determinar si existe evidencia empírica de persistencia en los rendimientos que permita capturar ganancias sistemáticas mediante una estrategia de tipo long-short.

---

## 📐 Métodos Utilizados

- **Estrategia Momentum 3x3**: clasificación mensual de acciones en carteras ganadoras (*long*) y perdedoras (*short*) según su retorno acumulado en los últimos 3 meses, observando su desempeño durante los 3 meses siguientes.
- **Retornos logarítmicos mensuales**: calculados como la suma de retornos logarítmicos diarios dentro de cada mes.
- **Análisis de persistencia**: evaluación de la probabilidad de que una acción clasificada como ganadora (o perdedora) en el período *t* se mantenga en dicha categoría en *t+1*, *t+2* y *t+3*.
- **Retorno de la estrategia long-short**: diferencia entre el retorno promedio de la cartera ganadora y la cartera perdedora.
- **Análisis descriptivo de carteras**: composición y rotación de las carteras construidas en cada período.

---

## 📊 Datos

- **Universo de acciones:** 15 acciones representativas del mercado colombiano seleccionadas por disponibilidad y calidad de datos.
- **Período:** mayo 2020 – mayo 2025 (60 observaciones mensuales por acción).
- **Fuente:** plataforma Economática.
- **Tratamiento de datos faltantes:** retorno nulo (log-return = 0) para períodos sin cotización.
- **Empresas incluidas:** BCOLOMBIA, BOGOTA, CEMARGOS, CORFICOLCF, ECOPETROL, GEB, GRUPOARGOS, GRUPOSURA, MINEROS, NUTRESA, PFAVAL, PFBCOLOM, PFCORFICOL, PFGRUPOARG, PFGRUPSURA.

---

## 🔍 Lo que se hizo

1. Se construyó una base de datos de retornos logarítmicos mensuales para las 15 acciones seleccionadas.
2. Se implementó la lógica de la estrategia 3x3: ranking por retorno acumulado en los últimos 3 meses, formación de carteras long (top) y short (bottom), y evaluación del retorno durante los 3 meses siguientes.
3. Se calculó la probabilidad de persistencia de cada acción en su categoría (ganadora/perdedora) para los períodos t+1, t+2 y t+3.
4. Se evaluó el retorno agregado de la estrategia long-short a lo largo del período analizado.
5. Se discutió la viabilidad del enfoque en un mercado emergente con características estructurales propias.

---

## 📈 Resultados

- Se observa **cierta persistencia en los rankings de rendimiento**, especialmente en acciones como CEMARGOS y GRUPOARGOS, con probabilidades de mantenerse en cartera long superiores al 60–70% hasta t+3.
- Sin embargo, el **retorno promedio de la estrategia long-short fue negativo**, lo que indica que la estrategia no logró capturar un diferencial de rentabilidad estadísticamente significativo en el período analizado.
- La probabilidad promedio de persistencia en la cartera long disminuye de 68.8% en t+1 a 49.1% en t+3, mientras que para la cartera short cae de 66.5% a 47.2%, lo que es consistente con una posible **reversión de rendimientos** en el corto plazo.
- Los resultados sugieren que en el mercado colombiano durante este período, el comportamiento de **reversión dominó sobre el efecto momentum clásico**, lo que podría interpretarse como evidencia a favor de la eficiencia de mercado en su forma débil.
- El estudio propone como extensión la segmentación de activos por sector o la incorporación de factores como volatilidad y volumen para mejorar la capacidad predictiva de la estrategia.

---

## 📄 Archivo

- [`momentum 3x3.pdf`](./momentum%203x3.pdf) — Artículo completo del estudio.
