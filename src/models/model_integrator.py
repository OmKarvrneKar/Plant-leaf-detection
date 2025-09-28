import os
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import json
from pathlib import Path

class ModelIntegrator:
    def __init__(self):
        """Detect and integrate available trained models"""
        self.available_models = self.detect_available_models()
        self.current_model = None
        self.current_processor = None
        
    def detect_available_models(self):
        """Detect all available trained models"""
        models = {}
        
        # Check for Transformers models
        model_dirs = ['my_leaf_model', 'plantvillage_model', 'real_plantvillage_model']
        
        for model_dir in model_dirs:
            if os.path.exists(model_dir):
                try:
                    # Check if it's a valid transformers model
                    config_path = os.path.join(model_dir, 'config.json')
                    if os.path.exists(config_path):
                        with open(config_path, 'r') as f:
                            config = json.load(f)
                        
                        models[model_dir] = {
                            'type': 'transformers',
                            'path': model_dir,
                            'classes': config.get('num_labels', 2),
                            'architecture': config.get('_name_or_path', 'unknown'),
                            'status': 'ready'
                        }
                        print(f"✅ Found Transformers model: {model_dir} ({config.get('num_labels', 2)} classes)")
                except Exception as e:
                    print(f"⚠️ Could not load {model_dir}: {e}")
        
        # Check for H5 models (Keras/TensorFlow)
        h5_files = ['my_trained_model.h5']
        for h5_file in h5_files:
            if os.path.exists(h5_file):
                models[h5_file] = {
                    'type': 'keras',
                    'path': h5_file,
                    'classes': 'unknown',
                    'architecture': 'keras_model',
                    'status': 'needs_conversion'
                }
                print(f"✅ Found Keras model: {h5_file}")
        
        return models
    
    def load_best_model(self):
        """Load the best available model"""
        if not self.available_models:
            print("❌ No trained models found!")
            return False
        
        # Priority: PlantVillage > my_leaf_model > others
        priority_order = ['plantvillage_model', 'real_plantvillage_model', 'my_leaf_model']
        
        selected_model = None
        for model_name in priority_order:
            if model_name in self.available_models and self.available_models[model_name]['type'] == 'transformers':
                selected_model = model_name
                break
        
        if not selected_model:
            # Use first available transformers model
            for model_name, info in self.available_models.items():
                if info['type'] == 'transformers':
                    selected_model = model_name
                    break
        
        if selected_model:
            return self.load_transformers_model(selected_model)
        else:
            print("⚠️ No compatible Transformers models found")
            return False
    
    def load_transformers_model(self, model_path):
        """Load a Transformers model"""
        try:
            print(f"🔄 Loading model: {model_path}")
            self.current_processor = AutoImageProcessor.from_pretrained(model_path)
            self.current_model = AutoModelForImageClassification.from_pretrained(model_path)
            self.current_model.eval()
            
            # Get model info
            config_path = os.path.join(model_path, 'config.json')
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            num_classes = config.get('num_labels', 2)
            id2label = config.get('id2label', {})
            
            print(f"✅ Model loaded successfully!")
            print(f"   📊 Classes: {num_classes}")
            print(f"   🏗️ Architecture: {config.get('_name_or_path', 'unknown')}")
            
            if id2label:
                print(f"   🏷️ Labels: {list(id2label.values())[:5]}{'...' if len(id2label) > 5 else ''}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load {model_path}: {e}")
            return False
    
    def get_model_info(self):
        """Get information about the current model"""
        if not self.current_model:
            return None
        
        config = self.current_model.config
        return {
            'num_classes': getattr(config, 'num_labels', 2),
            'id2label': getattr(config, 'id2label', {}),
            'label2id': getattr(config, 'label2id', {}),
            'architecture': getattr(config, '_name_or_path', 'unknown')
        }

def main():
    """Test model integration"""
    print("🔬 MODEL INTEGRATION SYSTEM")
    print("=" * 50)
    
    integrator = ModelIntegrator()
    
    print(f"\n📊 Available Models: {len(integrator.available_models)}")
    for name, info in integrator.available_models.items():
        print(f"   🤖 {name}: {info['classes']} classes ({info['type']})")
    
    print(f"\n🚀 Loading best model...")
    success = integrator.load_best_model()
    
    if success:
        model_info = integrator.get_model_info()
        print(f"\n✅ READY FOR USE!")
        print(f"   📊 Classes: {model_info['num_classes']}")
        print(f"   🎯 Model loaded and ready for predictions")
        
        return integrator
    else:
        print(f"\n❌ No compatible models found")
        return None

if __name__ == "__main__":
    main()