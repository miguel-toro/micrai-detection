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
        # Log input data info
        print(f"🔍 Iniciando preprocesamiento...")
        print(f"🔍 Tipo de image_data: {type(image_data)}")
        print(f"🔍 Longitud de image_data: {len(image_data) if image_data else 0}")
        
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
        import traceback
        print(f"❌ Error preprocessing image: {e}")
        print(f"❌ Error type: {type(e).__name__}")
        print(f"❌ Traceback: {traceback.format_exc()}")
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
    """Save image for future training - DISABLED on free tier ephemeral storage"""
    try:
        # Log the prediction but don't save to disk (ephemeral storage on free tier)
        predicted_class = "BLASTO" if prediction_result['clase'] == 1 else "NEGATIVE"
        confidence = prediction_result['confianza']
        print(f"📝 Predicción: {predicted_class} con {confidence*100:.2f}% confianza")
        print(f"ℹ️  Guardado de imágenes desactivado (free tier - almacenamiento efímero)")
        return None
        
    except Exception as e:
        print(f"Error en save_image_for_training: {e}")
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
        
        # Save image for training (disabled on free tier)
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
        import traceback
        import sys
        
        error_info = {
            'error_type': type(e).__name__,
            'error_message': str(e),
            'traceback': traceback.format_exc()
        }
        
        # Detailed error logging
        print("=" * 80)
        print("❌ ERROR COMPLETO EN /predict")
        print("=" * 80)
        print(f"Tipo de error: {error_info['error_type']}")
        print(f"Mensaje: {error_info['error_message']}")
        print(f"\nTraceback completo:")
        print(error_info['traceback'])
        print("=" * 80)
        
        return jsonify({
            'success': False, 
            'error': str(e),
            'error_type': type(e).__name__
        }), 500

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
            'total_images': total_images,
            'timestamp': datetime.now().isoformat(),
            'note': 'Free tier uses ephemeral storage - images do not persist'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'model_loaded': model is not None,
            'total_images': 0,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

# Debug endpoints
@app.route('/debug')
def debug():
    """Endpoint de diagnóstico completo"""
    import sys
    
    try:
        model_size = os.path.getsize(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
        model_size_mb = f"{model_size / (1024*1024):.2f} MB" if model_size else 'N/A'
    except Exception as e:
        model_size_mb = f'Error: {e}'
    
    debug_info = {
        'status': 'ok',
        'python_version': sys.version,
        'tensorflow_version': tf.__version__,
        'numpy_version': np.__version__,
        'working_directory': os.getcwd(),
        'model_path': MODEL_PATH,
        'model_exists': os.path.exists(MODEL_PATH),
        'model_size': model_size_mb,
        'model_loaded': model is not None,
        'model_config': MODEL_CONFIG,
        'upload_folder': UPLOAD_FOLDER,
        'uploads_exists': os.path.exists(UPLOAD_FOLDER),
        'environment': {
            'PORT': os.environ.get('PORT', 'Not set'),
            'PYTHON_VERSION': os.environ.get('PYTHON_VERSION', 'Not set')
        }
    }
    
    return jsonify(debug_info)

@app.route('/test-model')
def test_model():
    """Test rápido del modelo con imagen aleatoria"""
    if model is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded'
        }), 500
    
    try:
        import time
        
        print("🧪 Iniciando test del modelo...")
        
        # Crear imagen de prueba 224x224 RGB
        test_img = np.random.rand(1, 224, 224, 3).astype(np.float32)
        print(f"🧪 Imagen de prueba creada: shape={test_img.shape}, dtype={test_img.dtype}")
        
        start_time = time.time()
        prediction = model.predict(test_img, verbose=0)
        inference_time = time.time() - start_time
        
        print(f"✅ Predicción exitosa: {prediction[0][0]:.6f} en {inference_time:.3f}s")
        
        return jsonify({
            'success': True,
            'prediction': float(prediction[0][0]),
            'inference_time_seconds': round(inference_time, 3),
            'message': '✅ Model is working correctly!',
            'test_image_shape': list(test_img.shape)
        })
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ Error en test del modelo: {e}")
        print(f"❌ Traceback: {error_details}")
        return jsonify({
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__,
            'traceback': error_details
        }), 500

@app.route('/test-preprocess')
def test_preprocess():
    """Test del preprocesamiento de imagen"""
    try:
        # Imagen de prueba 1x1 pixel en base64
        test_base64 = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
        
        print("🧪 Testing preprocesar_imagen...")
        result = preprocesar_imagen(test_base64, mostrar_pasos=True)
        
        if result is not None:
            return jsonify({
                'success': True,
                'processed_shape': list(result.shape),
                'processed_dtype': str(result.dtype),
                'processed_min': float(result.min()),
                'processed_max': float(result.max()),
                'message': '✅ Preprocessing working!'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'preprocesar_imagen returned None'
            }), 500
            
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
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