# Sistema de Detección de Neumonía mediante Visión Transformer

**Curso:** Inteligencia Artificial, Principios y Técnicas

**Autores:**
- Chávez Castillo Jorge
- Sandoval Vargas Robert
- Carranza Castillo Cristian David
- Silvestre Miguel Alex Paul

**Docente:** Sagastegui Chigne Teobaldo Hernan

## Descripción
Sistema web de detección de neumonía en radiografías de tórax 
mediante un modelo Vision Transformer (ViT) integrado desde Hugging Face.

## Modelo de IA
- Modelo: dima806/chest_xray_pneumonia_detection
- Arquitectura: Vision Transformer (ViT)
- Base: google/vit-base-patch16-224-in21k
- Accuracy: 96.08%
- F1-Score: 0.9608

## Tecnologías
- Python 3.10
- Streamlit
- Hugging Face Transformers
- PyTorch
- Plotly

## Dataset
- Chest X-Ray Images — Kermany et al. (2018)
- Fuente: Kaggle
- Total: 5,863 radiografías
- Clases: NORMAL / PNEUMONIA

## Métricas
| Clase | Precisión | Recall | F1-Score |
|-------|-----------|--------|----------|
| NORMAL | 0.9603 | 0.9614 | 0.9608 |
| PNEUMONIA | 0.9614 | 0.9602 | 0.9608 |
| Accuracy | | | 96.08% |

## Instalación y Ejecución
1. Descarga o clona el repositorio
2. Instala las dependencias:
pip install -r requirements.txt
3. Ejecuta la app:
streamlit run app.py
4. Abre en el navegador:
http://localhost:8501
