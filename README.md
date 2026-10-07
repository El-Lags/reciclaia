# ♻️ ReciclaIA · clasificador de residuos con IA

App web que clasifica la foto de un residuo (cartón, vidrio, metal, papel, plástico o basura), muestra la confianza de cada clase y genera una recomendación de reciclaje con un modelo de lenguaje (Gemini).

**Proyecto grupal · Certamen · IA Generativa para la Gestión Industrial · UDD**

🔗 **App publicada:** _[pegar aquí el link de Streamlit Cloud]_

## Integrantes
- Felipe Alonso
- Sofía Fariña
- Vicente Lagunas

## Flujo
📷 Imagen → 🧠 EfficientNet-B0 (clase + confianza) → 📊 Resultado visual → 💬 Gemini → ✅ Recomendación

## Problema
Separar mal los residuos contamina lotes completos de material reciclable. La app ayuda a identificar el material y orienta sobre cómo y dónde reciclarlo, con aviso de revisión humana cuando el modelo no está seguro.

## Datos
- **Dataset:** TrashNet (Thung y Yang, Stanford, 2016), archivo `dataset-resized.zip` del repositorio de Hugging Face [`garythung/trashnet`](https://huggingface.co/datasets/garythung/trashnet) (2.527 fotos; el repositorio también trae la versión original de las mismas fotos, que no se usa para no duplicarlas).
- **Licencia:** MIT.
- **Clases:** cardboard, glass, metal, paper, plastic, trash.
- **Imágenes por clase (2.527 en total):** cardboard 403 · glass 501 · metal 410 · paper 594 · plastic 482 · trash 137. La clase `trash` es solo el 5,4 % (desbalance de 4,3 a 1 entre la mayor y la menor).
- **Duplicados:** se eliminaron 3 copias exactas, quedan 2.524 imágenes.
- **División estratificada (semilla 1993):**

| Clase | Train | Val | Test |
|---|---|---|---|
| cardboard | 282 | 61 | 60 |
| glass | 350 | 75 | 76 |
| metal | 286 | 62 | 61 |
| paper | 416 | 89 | 89 |
| plastic | 336 | 72 | 72 |
| trash | 96 | 20 | 21 |
| **Total** | **1.766** | **379** | **379** |

- **Limitaciones:** una sola fuente, fondos lisos, clase `trash` escasa, posibles casi-duplicados. Detalle en el notebook.
- El dataset **no** está en este repositorio por su tamaño; se descarga desde el enlace.

## Modelo
| Modelo | Accuracy (test) | Accuracy balanceada (test) |
|---|---|---|
| CLIP zero-shot (línea base) | 66,0 % | 59,9 % |
| **EfficientNet-B0 (timm), transfer learning** | **83,6 %** | **82,6 %** |

Elegimos EfficientNet-B0 porque supera a CLIP por 17,6 puntos de accuracy y 22,7 de accuracy balanceada con las mismas 379 imágenes de prueba, y además pesa unos 16 MB, cabe en GitHub y corre en CPU. La clase más difícil es `trash` (precision 0,41 y recall 0,71 con solo 21 imágenes de prueba).

**Error más costoso:** clasificar como reciclable un residuo no reciclable, porque contamina el lote completo. Ocurrió con 6 de las 21 imágenes `trash` (28,6 %). El error inverso también cuesta, porque se pierde material reciclable: 22 imágenes reciclables se clasificaron como `trash` (11 papel, 6 plástico y 5 metal).

## Estructura
```
app.py                       aplicación Streamlit
requirements.txt             librerías
notebooks/entrenamiento.ipynb  datos, comparación de modelos y métricas
modelo/                      pesos (.pt), clases y métricas del test
ejemplos/                    imágenes de prueba
.streamlit/config.toml       tema de la app
```

## Cómo ejecutarla
```bash
pip install -r requirements.txt
# clave de Gemini (nunca se sube a GitHub)
mkdir .streamlit   # si no existe
echo GEMINI_API_KEY = "tu_clave" > .streamlit/secrets.toml
streamlit run app.py
```
En Streamlit Cloud la clave se guarda en **Advanced settings → Secrets** como `GEMINI_API_KEY = "..."`.

## Uso de IA
- **Modelos de visión:** EfficientNet-B0 (timm) ajustado con TrashNet y CLIP (OpenAI) como línea base.
- **IA generativa en la app:** Google Gemini genera la recomendación a partir de la clase y la confianza (solo texto, no recibe la imagen).
- **Asistentes de IA para programar:** _[declarar aquí qué herramientas usaron y para qué, por ejemplo Claude Code para estructurar el código de la app y el notebook]_.

## Consideraciones éticas
- Prototipo **educativo**: no reemplaza las indicaciones de la municipalidad ni de un punto limpio.
- Con confianza menor a 70 % la app pide revisión humana. En el test, las fotos con confianza ≥ 70 % (89 % del total) tienen 89,0 % de accuracy; las demás se derivan a revisión.
- No se usan fotos de personas ni datos sensibles.
- Sesgo: el dataset proviene de productos de EE. UU. con fondos controlados; puede rendir peor con envases chilenos o fondos complejos.
