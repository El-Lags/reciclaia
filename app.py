"""ReciclaIA · clasificador de residuos con IA.

Flujo: imagen -> EfficientNet-B0 (clase + confianza) -> resultado visual -> LLM (Gemini) -> recomendación.
"""
import json
import os

import requests
import streamlit as st
import timm
import torch
from PIL import Image
from torchvision import transforms

# ---------------------------------------------------------------- configuración
st.set_page_config(page_title="ReciclaIA", page_icon="♻️", layout="wide")

RAIZ = os.path.dirname(os.path.abspath(__file__))
RUTA_PESOS = os.path.join(RAIZ, "modelo", "modelo_efficientnet.pt")
RUTA_CLASES = os.path.join(RAIZ, "modelo", "clases.json")
RUTA_METRICAS = os.path.join(RAIZ, "modelo", "metricas.json")
CARPETA_EJEMPLOS = os.path.join(RAIZ, "ejemplos")

UMBRAL_CONFIANZA = 0.70  # bajo este valor se pide revisión humana (ver sección 6.1 del notebook)

# nombre en español, ícono y contenedor sugerido (Chile) para cada clase del dataset
INFO_CLASES = {
    "cardboard": ("Cartón", "📦", "contenedor azul (papel y cartón)"),
    "glass": ("Vidrio", "🍾", "campana verde de vidrio"),
    "metal": ("Metal", "🥫", "contenedor amarillo (envases y latas)"),
    "paper": ("Papel", "📄", "contenedor azul (papel y cartón)"),
    "plastic": ("Plástico", "🧴", "contenedor amarillo (envases plásticos)"),
    "trash": ("Basura no reciclable", "🗑️", "basura general"),
}

# ---------------------------------------------------------------- estilo
st.markdown(
    """
    <style>
    .titulo {font-size: 2.4rem; font-weight: 800; color: #1b5e20; margin-bottom: 0;}
    .subtitulo {color: #4e6b50; margin-top: 0; font-size: 1.05rem;}
    .tarjeta {background: #ffffff; border: 1px solid #c8e6c9; border-left: 6px solid #2e7d32;
              border-radius: 10px; padding: 1rem 1.2rem; margin-bottom: 0.8rem;}
    .clase {font-size: 1.8rem; font-weight: 700; color: #1b5e20;}
    .aviso {background: #fff8e1; border-left: 6px solid #f9a825; border-radius: 8px;
            padding: 0.7rem 1rem; margin: 0.6rem 0;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- modelo
@st.cache_resource(show_spinner="Cargando el modelo…")
def cargar_modelo():
    """Carga EfficientNet-B0 con los pesos entrenados en el notebook."""
    with open(RUTA_CLASES, encoding="utf-8") as f:
        clases = json.load(f)
    modelo = timm.create_model("efficientnet_b0", pretrained=False, num_classes=len(clases))
    modelo.load_state_dict(torch.load(RUTA_PESOS, map_location="cpu"))
    modelo.eval()
    return modelo, clases


PREPROCESO = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # estadísticas de ImageNet
])


def clasificar(imagen, modelo, clases):
    """Devuelve un diccionario {clase: probabilidad} ordenado de mayor a menor."""
    x = PREPROCESO(imagen.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(modelo(x), dim=1)[0].tolist()
    return dict(sorted(zip(clases, probs), key=lambda par: -par[1]))


def cargar_metricas():
    """Métricas medidas en el conjunto de prueba (las guarda el notebook)."""
    try:
        with open(RUTA_METRICAS, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


# ---------------------------------------------------------------- LLM (Gemini)
def pedir_recomendacion(clase, confianza):
    """Envía solo texto (clase y confianza) a Gemini y devuelve la recomendación."""
    try:                                   # si no existe secrets.toml, Streamlit lanza un error
        secretos = dict(st.secrets)
    except Exception:
        secretos = {}
    clave = secretos.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not clave:
        return "⚠️ Falta configurar GEMINI_API_KEY como secreto de la aplicación."
    modelo_llm = secretos.get("GEMINI_MODEL") or os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")

    nombre, _, contenedor = INFO_CLASES.get(clase, (clase, "", "según normativa local"))
    prompt = f"""Eres un asistente educativo sobre reciclaje domiciliario en Chile.
Un modelo de visión clasificó la foto de un residuo así:
- Clase detectada: {nombre}
- Confianza del modelo: {confianza:.0%}
- Destino habitual: {contenedor}

Responde en español, con un máximo de 120 palabras, usando estas secciones:
1. Cómo preparar el residuo antes de reciclarlo.
2. Dónde depositarlo.
3. Un dato breve sobre su impacto ambiental.

Reglas:
- No saludes ni te presentes: empieza directamente con la sección 1.
- Háblale solo de reciclaje y manejo de residuos; si algo no corresponde, no lo inventes.
- {"La confianza es BAJA: empieza diciendo que la clasificación es incierta y pide al usuario verificar el material a simple vista o consultar la norma local antes de decidir." if confianza < UMBRAL_CONFIANZA else "Si el material podría ser otro, sugiere verificarlo."}
- Es información educativa: no reemplaza las indicaciones de tu municipalidad ni de un punto limpio."""

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{modelo_llm}:generateContent"
    try:
        for intento in range(2):  # un reintento si Google está lento o saturado
            try:
                r = requests.post(
                    url,
                    headers={"x-goog-api-key": clave, "Content-Type": "application/json"},
                    json={"contents": [{"parts": [{"text": prompt}]}]},
                    timeout=45,
                )
                if r.status_code in (429, 500, 503) and intento == 0:
                    continue
                break
            except requests.Timeout:
                if intento == 1:
                    raise
        r.raise_for_status()
        partes = r.json()["candidates"][0]["content"]["parts"]
        return "".join(p["text"] for p in partes if "text" in p and not p.get("thought"))
    except requests.HTTPError as e:  # Google respondió con un error: mostramos su código y mensaje
        try:
            detalle = e.response.json()["error"]["message"][:200]
        except Exception:
            detalle = e.response.text[:200]
        return f"⚠️ No se pudo obtener la recomendación (error {e.response.status_code}): {detalle}"
    except Exception as e:  # la app no debe caerse si el LLM falla
        return f"⚠️ No se pudo obtener la recomendación ({type(e).__name__}). Intenta de nuevo."


# ---------------------------------------------------------------- barra lateral
with st.sidebar:
    st.markdown("## ♻️ ReciclaIA")
    st.write("Sube la foto de un residuo y la IA te dice **de qué material es** y **dónde reciclarlo**.")
    st.markdown("---")
    st.markdown("### Cómo funciona")
    st.markdown(
        "1. 📷 Subes una imagen\n"
        "2. 🧠 EfficientNet-B0 la clasifica\n"
        "3. 📊 Ves la confianza de cada clase\n"
        "4. 💬 Gemini te da una recomendación"
    )
    st.markdown("---")
    st.caption("Modelo: EfficientNet-B0 (timm) ajustado con TrashNet · Proyecto del ramo IA Generativa para la Gestión Industrial, UDD.")

# ---------------------------------------------------------------- encabezado
st.markdown('<p class="titulo">♻️ ReciclaIA</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitulo">Clasifica residuos con visión por computador y recibe una guía de reciclaje generada por IA.</p>', unsafe_allow_html=True)
st.markdown(
    '<div class="aviso">🎓 <b>Prototipo educativo.</b> Los resultados pueden equivocarse; '
    'ante la duda, consulta las indicaciones de tu municipalidad o punto limpio.</div>',
    unsafe_allow_html=True,
)

try:
    modelo, clases = cargar_modelo()
except Exception as e:
    st.error(f"No se pudo cargar el modelo ({type(e).__name__}). Revisa que la carpeta `modelo/` tenga los pesos y las clases.")
    st.stop()

pestana_app, pestana_modelo = st.tabs(["📷 Clasificar", "📊 Sobre el modelo"])

# ---------------------------------------------------------------- pestaña 1: clasificar
with pestana_app:
    col_entrada, col_resultado = st.columns(2, gap="large")

    with col_entrada:
        st.subheader("1. Elige una imagen")
        archivo = st.file_uploader("Sube una foto (JPG o PNG)", type=["jpg", "jpeg", "png", "webp"])

        ejemplos = sorted(
            f for f in (os.listdir(CARPETA_EJEMPLOS) if os.path.isdir(CARPETA_EJEMPLOS) else [])
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
        )
        ejemplo = st.selectbox("…o prueba con un ejemplo", ["—"] + ejemplos)

        imagen = None
        if archivo is not None:
            imagen = Image.open(archivo)
        elif ejemplo != "—":
            imagen = Image.open(os.path.join(CARPETA_EJEMPLOS, ejemplo))
        if imagen is not None:
            st.image(imagen, caption="Imagen a clasificar", width="stretch")

    with col_resultado:
        st.subheader("2. Resultado")
        if imagen is None:
            st.info("Sube una imagen o elige un ejemplo para empezar.")
        else:
            probs = clasificar(imagen, modelo, clases)
            clase, confianza = next(iter(probs.items()))
            nombre, icono, contenedor = INFO_CLASES.get(clase, (clase, "❓", "—"))

            st.markdown(
                f'<div class="tarjeta"><div class="clase">{icono} {nombre}</div>'
                f"Confianza de <b>esta imagen</b>: <b>{confianza:.1%}</b><br>"
                f"Destino habitual: {contenedor}</div>",
                unsafe_allow_html=True,
            )
            if confianza < UMBRAL_CONFIANZA:
                st.warning(
                    "⚠️ **Confianza baja.** El modelo no está seguro: revisa el material a simple vista "
                    "(revisión humana) antes de decidir dónde depositarlo."
                )

            st.markdown("**Probabilidad de cada clase**")
            st.bar_chart(
                {INFO_CLASES.get(c, (c,))[0]: p for c, p in probs.items()},
                horizontal=True,
                color="#2e7d32",
            )

            st.markdown("---")
            st.subheader("3. Recomendación con IA")
            if st.button("💬 Generar recomendación", type="primary"):
                with st.spinner("Consultando a Gemini…"):
                    st.session_state["recomendacion"] = pedir_recomendacion(clase, confianza)
            if "recomendacion" in st.session_state:
                st.markdown(st.session_state["recomendacion"])
                st.caption("Generado por IA. Es orientación educativa y puede contener errores.")

# ---------------------------------------------------------------- pestaña 2: sobre el modelo
with pestana_modelo:
    st.subheader("Desempeño del modelo en el conjunto de prueba")
    st.caption(
        "Estas cifras se midieron **una sola vez** con imágenes que el modelo nunca vio al entrenar. "
        "No son la confianza de la foto que subiste: esa es distinta para cada imagen."
    )
    metricas = cargar_metricas()
    if metricas is None:
        st.info("Aún no hay métricas: ejecuta el notebook y copia `modelo/metricas.json` al repositorio.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{metricas['accuracy']:.1%}")
        c2.metric("Accuracy balanceada", f"{metricas['accuracy_balanceada']:.1%}")
        c3.metric("Imágenes de prueba", metricas["n_test"])
        st.markdown("**Recall por clase** (qué porcentaje de cada material detecta el modelo)")
        st.bar_chart(
            {INFO_CLASES.get(c, (c,))[0]: v for c, v in metricas["recall_por_clase"].items()},
            horizontal=True,
            color="#2e7d32",
        )
        st.markdown(
            f"**Línea base:** CLIP sin entrenar obtiene {metricas['clip_accuracy']:.1%} de accuracy "
            "con las mismas imágenes de prueba."
        )

    st.markdown("### Limitaciones y ética")
    st.markdown(
        "- El dataset (TrashNet) tiene fondos simples y una sola fuente: con fondos complejos o mala luz el modelo puede fallar.\n"
        "- La clase *Basura* tiene pocas imágenes, así que es la más difícil de reconocer.\n"
        "- Equivocarse en un residuo contaminado (p. ej. mandar basura al contenedor de reciclaje) **contamina** todo el lote: por eso se pide revisión humana cuando la confianza es baja.\n"
        "- La recomendación la escribe un modelo de lenguaje a partir de la clase y la confianza; no ve la imagen."
    )
