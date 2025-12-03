from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import os
from datetime import datetime
import uuid

app = Flask(__name__)
CORS(app)

# Configuration - using your existing uploaded model
UPLOAD_FOLDER = 'uploads'
MODEL_PATH = 'models/blastocystis_model.h5'  # Your uploaded model
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Model configuration - matching your Colab v2 settings (224x224 RGB)
MODEL_CONFIG = {
    "tamaño": (224, 224),
    "canales": 3,  # RGB
    "descripcion": "Modelo v2 - 224x224 RGB",
    "umbral_deteccion": 0.7  # 70% threshold like Colab
}

# Create folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('models', exist_ok=True)

# Load your existing model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Modelo cargado exitosamente!")
    print(f"🎯 Configuración: {MODEL_CONFIG['descripcion']}")
    print(f"📐 Input shape: {model.input_shape}")
    print(f"📊 Output shape: {model.output_shape}")
except Exception as e:
    print(f"⚠️ Error loading model: {e}")
    print("🔄 Ejecutando en modo demo")
    model = None

def preprocesar_imagen(image_data, mostrar_pasos=False):
    """Preprocess image exactly like your Colab - 224x224 RGB"""
    
    tamaño_objetivo = MODEL_CONFIG['tamaño']
    canales_requeridos = MODEL_CONFIG['canales']
    
    if mostrar_pasos:
        print(f"🎯 Modelo requiere: {tamaño_objetivo} con {canales_requeridos} canales (RGB)")
    
    try:
        # Decode base64 image
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        imagen = Image.open(io.BytesIO(image_bytes))
        
        if mostrar_pasos:
            print(f"📐 Tamaño original: {imagen.size}")
            print(f"🎨 Modo original: {imagen.mode}")
        
        # Resize to 224x224
        imagen_redimensionada = imagen.resize(tamaño_objetivo, Image.Resampling.LANCZOS)
        
        # Convert to RGB (3 channels)
        if imagen_redimensionada.mode != 'RGB':
            imagen_procesada = imagen_redimensionada.convert('RGB')
            if mostrar_pasos:
                print("🌈 Convertido a RGB")
        else:
            imagen_procesada = imagen_redimensionada
            if mostrar_pasos:
                print("🌈 Ya está en RGB")
        
        # Convert to numpy array
        img_array = np.array(imagen_procesada)
        
        # Normalize (0-255 -> 0-1)
        img_array = img_array.astype(np.float32) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        if mostrar_pasos:
            print(f"🔢 Shape del array: {img_array.shape}")
            print(f"⚖️ Normalizado: rango [{img_array.min():.3f}, {img_array.max():.3f}]")
            print("✅ Preprocesamiento completado!")
        
        return img_array
        
    except Exception as e:
        print(f"❌ Error preprocessing image: {e}")
        return None

def clasificar_blastocystis(imagen_preprocesada, umbral_deteccion=0.7, mostrar_detalles=False):
    """Classify exactly like your Colab with 70% threshold"""
    
    if mostrar_detalles:
        print("🧠 Ejecutando predicción...")
    
    # Make prediction
    prediccion_raw = model.predict(imagen_preprocesada, verbose=0)
    probabilidad = float(prediccion_raw[0][0])
    
    if mostrar_detalles:
        print(f"📊 Probabilidad bruta: {probabilidad:.6f}")
        print(f"🎯 Umbral de detección: {umbral_deteccion:.1f}")
    
    # Determine class and confidence with 70% threshold (exactly like Colab)
    if probabilidad > umbral_deteccion:
        clase = 1  # Blastocystis
        etiqueta_es = "BLASTOCYSTIS DETECTADO"
        etiqueta_en = "BLASTOCYSTIS DETECTED"
        confianza = probabilidad
        emoji = "🔴"
    else:
        clase = 0  # Negative
        etiqueta_es = "NO ES BLASTOCYSTIS"
        etiqueta_en = "NOT BLASTOCYSTIS"
        confianza = 1 - probabilidad
        emoji = "🟢"
    
    # Determine confidence level exactly like Colab
    if confianza >= 0.9:
        nivel_confianza_es = "MUY ALTA"
        nivel_confianza_en = "VERY HIGH"
    elif confianza >= 0.7:
        nivel_confianza_es = "ALTA"
        nivel_confianza_en = "HIGH"
    elif confianza >= 0.6:
        nivel_confianza_es = "MODERADA"
        nivel_confianza_en = "MODERATE"
    else:
        nivel_confianza_es = "BAJA"
        nivel_confianza_en = "LOW"
    
    # Generate interpretation exactly like Colab
    if clase == 1:
        interpretacion_es = "✅ El modelo detectó características de Blastocystis con ALTA CONFIANZA (>70%)"
        interpretacion_en = "✅ Model detected Blastocystis characteristics with HIGH CONFIDENCE (>70%)"
        detalle_es = "🔬 Se observaron patrones MUY consistentes con formas vacuolares"
        detalle_en = "🔬 Very consistent patterns with vacuolar forms observed"
    else:
        if probabilidad > 0.5:
            interpretacion_es = "⚠️ El modelo detectó algunas características de Blastocystis, pero con confianza INSUFICIENTE (<70%)"
            interpretacion_en = "⚠️ Model detected some Blastocystis characteristics, but with INSUFFICIENT confidence (<70%)"
            detalle_es = "🔬 Se requiere mayor certeza para confirmar la presencia del parásito"
            detalle_en = "🔬 Greater certainty required to confirm parasite presence"
        else:
            interpretacion_es = "❌ El modelo NO detectó formas vacuolares de Blastocystis"
            interpretacion_en = "❌ Model did NOT detect Blastocystis vacuolar forms"
            detalle_es = "🔬 La imagen no presenta características del parásito objetivo"
            detalle_en = "🔬 Image does not present target parasite characteristics"
    
    resultados = {
        'probabilidad': probabilidad,
        'clase': clase,
        'etiqueta_es': etiqueta_es,
        'etiqueta_en': etiqueta_en,
        'confianza': confianza,
        'confianza_porcentaje': confianza * 100,
        'emoji': emoji,
        'nivel_confianza_es': nivel_confianza_es,
        'nivel_confianza_en': nivel_confianza_en,
        'interpretacion_es': interpretacion_es,
        'interpretacion_en': interpretacion_en,
        'detalle_es': detalle_es,
        'detalle_en': detalle_en,
        'umbral_usado': umbral_deteccion,
        'modelo_usado': MODEL_CONFIG['descripcion']
    }
    
    if mostrar_detalles:
        print(f"✅ Predicción completada: {etiqueta_es}")
        print(f"📈 Confianza: {confianza * 100:.2f}% ({nivel_confianza_es})")
        print(f"🔍 {interpretacion_es}")
        print(f"💡 {detalle_es}")
    
    return resultados

def save_image_for_training(image_data, prediction_result):
    """Save image for future training with Colab-style naming"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Create filename with prediction info (Colab style)
        confidence = prediction_result['confianza']
        predicted_class = "BLASTO" if prediction_result['clase'] == 1 else "NEGATIVE"
        
        filename = f"{timestamp}_{unique_id}_{predicted_class}_conf{confidence:.2f}.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        
        image.save(filepath, 'JPEG', quality=95)
        print(f"✅ Imagen guardada para entrenamiento: {filename}")
        return filename
        
    except Exception as e:
        print(f"Error saving image: {e}")
        return None

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'success': False, 'error': 'No image data provided'}), 400
        
        image_data = data['image']
        language = data.get('language', 'es')
        
        if model is None:
            # Demo mode with Colab-style results
            print("📝 Ejecutando en modo demo (modelo no encontrado)")
            resultado = {
                'probabilidad': 0.87,
                'clase': 1,
                'etiqueta_es': "BLASTOCYSTIS DETECTADO",
                'etiqueta_en': "BLASTOCYSTIS DETECTED", 
                'confianza': 0.87,
                'confianza_porcentaje': 87.0,
                'emoji': "🔴",
                'nivel_confianza_es': "ALTA",
                'nivel_confianza_en': "HIGH",
                'interpretacion_es': "✅ El modelo detectó características de Blastocystis con ALTA CONFIANZA (>70%)",
                'interpretacion_en': "✅ Model detected Blastocystis characteristics with HIGH CONFIDENCE (>70%)",
                'detalle_es': "🔬 Se observaron patrones MUY consistentes con formas vacuolares",
                'detalle_en': "🔬 Very consistent patterns with vacuolar forms observed",
                'umbral_usado': 0.7,
                'modelo_usado': "Demo Mode - Modelo v2 simulado"
            }
        else:
            # Real AI prediction using your uploaded model
            print("🤖 Ejecutando predicción con modelo real...")
            processed_image = preprocesar_imagen(image_data, mostrar_pasos=True)
            
            if processed_image is None:
                return jsonify({'success': False, 'error': 'Failed to process image'}), 400
            
            resultado = clasificar_blastocystis(processed_image, umbral_deteccion=0.7, mostrar_detalles=True)
        
        # Format response for frontend (compatible with original format)
        response = {
            'success': True,
            'predictions': [
                {
                    'label': resultado['etiqueta_en'] if language == 'en' else resultado['etiqueta_es'],
                    'label_es': resultado['etiqueta_es'],
                    'label_en': resultado['etiqueta_en'],
                    'confidence': resultado['confianza']
                },
                {
                    'label': "NO ES BLASTOCYSTIS" if resultado['clase'] == 1 else "BLASTOCYSTIS DETECTADO",
                    'label_es': "NO ES BLASTOCYSTIS" if resultado['clase'] == 1 else "BLASTOCYSTIS DETECTADO",
                    'label_en': "NOT BLASTOCYSTIS" if resultado['clase'] == 1 else "BLASTOCYSTIS DETECTED",
                    'confidence': 1 - resultado['confianza']
                }
            ],
            'detailed_analysis': {
                'probabilidad_bruta': resultado['probabilidad'],
                'umbral_decision': resultado['umbral_usado'],
                'nivel_confianza': resultado['nivel_confianza_es'] if language == 'es' else resultado['nivel_confianza_en'],
                'interpretacion': resultado['interpretacion_es'] if language == 'es' else resultado['interpretacion_en'],
                'detalle': resultado['detalle_es'] if language == 'es' else resultado['detalle_en'],
                'modelo_usado': resultado['modelo_usado'],
                'emoji': resultado['emoji'],
                'confianza_porcentaje': resultado['confianza_porcentaje']
            },
            'timestamp': datetime.now().isoformat(),
            'mode': 'demo' if model is None else 'ai'
        }
        
        # Save image for training
        saved_filename = save_image_for_training(image_data, resultado)
        if saved_filename:
            response['saved_image'] = saved_filename
        
        # Console output (like Colab)
        print(f"{'='*50}")
        print("📋 ANÁLISIS DETALLADO:")
        print(f"{'='*50}")
        print(f"🤖 Modelo: {resultado['modelo_usado']}")
        print(f"📊 Probabilidad bruta: {resultado['probabilidad']:.6f}")
        print(f"🎯 Umbral de decisión: {resultado['umbral_usado']:.1f} (70%)")
        print(f"🏷️ Clase predicha: {resultado['etiqueta_es']}")
        print(f"📈 Confianza: {resultado['confianza_porcentaje']:.2f}% ({resultado['nivel_confianza_es']})")
        print(f"🔍 {resultado['interpretacion_es']}")
        print(f"💡 {resultado['detalle_es']}")
        print(f"{'='*50}")
        
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Error en predicción: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health')
def health_check():
    try:
        # Count files in uploads folder
        if os.path.exists(UPLOAD_FOLDER):
            # Count only image files
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            total_images = len([f for f in os.listdir(UPLOAD_FOLDER) 
                              if os.path.splitext(f.lower())[1] in image_extensions])
        else:
            total_images = 0
            
        return jsonify({
            'status': 'healthy',
            'model_loaded': model is not None,
            'model_config': MODEL_CONFIG,
            'upload_folder': UPLOAD_FOLDER,
            'total_images': total_images,  # This is what the frontend needs
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'model_loaded': model is not None,
            'total_images': 0,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    
    print("🚀 Sistema de Detección de Blastocystis")
    print(f"🤖 Configuración: {MODEL_CONFIG['descripcion']}")
    print(f"🎯 Umbral de detección: {MODEL_CONFIG['umbral_deteccion']} (70%)")
    print(f"📁 Modelo: {MODEL_PATH}")
    print(f"✅ Modelo cargado: {'Sí' if model is not None else 'No (modo demo)'}")
    print(f"🌐 Puerto: {port}")
    
    # Production mode - no debug
    app.run(host='0.0.0.0', port=port, debug=False)
