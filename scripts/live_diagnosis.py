import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import time
import os
from datetime import datetime

class FixedLiveDiagnosis:
    def __init__(self, model_path="./my_leaf_model"):
        """Initialize the working live diagnosis system"""
        print("🔧 FIXED Live Plant Disease Detection")
        print("=" * 50)
        
        try:
            print(f"🔄 Loading model: {model_path}")
            self.processor = AutoImageProcessor.from_pretrained(model_path)
            self.model = AutoModelForImageClassification.from_pretrained(model_path)
            self.model.eval()
            
            # Get model info
            config = self.model.config
            self.num_classes = getattr(config, 'num_labels', 2)
            self.id2label = getattr(config, 'id2label', {0: "diseased", 1: "healthy"})
            self.label2id = getattr(config, 'label2id', {"diseased": 0, "healthy": 1})
            
            print(f"✅ Model loaded successfully!")
            print(f"   📊 Classes: {self.num_classes}")
            print(f"   🏷️ Labels: {list(self.id2label.values())}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
        
        # Create results directory
        self.save_dir = "live_diagnosis_results"
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Disease information based on model type
        self.setup_disease_info()
    
    def setup_disease_info(self):
        """Setup disease information based on model"""
        if self.num_classes == 2:
            # Binary classification
            self.disease_info = {
                "diseased": {
                    "status": "🔴 DISEASE DETECTED",
                    "description": "Plant leaf shows signs of disease",
                    "actions": [
                        "🔍 Examine leaf closely for spots, discoloration, or damage",
                        "✂️ Remove affected leaves with clean scissors",
                        "🧴 Apply appropriate treatment (fungicide/bactericide)",
                        "💧 Adjust watering - avoid wetting leaves",
                        "🌬️ Improve air circulation around plant"
                    ]
                },
                "healthy": {
                    "status": "🟢 HEALTHY LEAF",
                    "description": "Plant leaf appears healthy",
                    "actions": [
                        "✅ Continue current care routine",
                        "👀 Monitor regularly for any changes",
                        "💧 Maintain proper watering schedule",
                        "🌱 Ensure adequate nutrition",
                        "☀️ Provide appropriate light conditions"
                    ]
                }
            }
        else:
            # Multi-class classification
            self.disease_info = {}
            for class_id, class_name in self.id2label.items():
                if 'healthy' in class_name.lower():
                    self.disease_info[class_name] = {
                        "status": f"🟢 {class_name.upper()}",
                        "description": f"Healthy {class_name.split('___')[0] if '___' in class_name else 'plant'}",
                        "actions": ["Continue care", "Regular monitoring"]
                    }
                else:
                    parts = class_name.split('___') if '___' in class_name else [class_name]
                    plant = parts[0] if parts else "Plant"
                    disease = parts[1] if len(parts) > 1 else "Disease"
                    
                    self.disease_info[class_name] = {
                        "status": f"🔴 {disease.upper()}",
                        "description": f"{disease} detected in {plant}",
                        "actions": [
                            f"🧴 Apply specific treatment for {disease}",
                            f"✂️ Remove affected {plant} parts",
                            "🔬 Consult agricultural expert if needed"
                        ]
                    }
    
    def predict_disease(self, image):
        """Predict disease with confidence"""
        try:
            inputs = self.processor(images=image, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            predicted_class_id = probabilities.argmax().item()
            confidence = probabilities[0][predicted_class_id].item()
            predicted_label = self.id2label[predicted_class_id]
            
            # Get all class probabilities for analysis
            all_probs = {}
            for class_id, class_name in self.id2label.items():
                prob = probabilities[0][int(class_id)].item()
                all_probs[class_name] = prob
            
            return predicted_label, confidence, all_probs
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return "unknown", 0.0, {}
    
    def analyze_leaf_visual(self, frame):
        """Quick visual analysis of the leaf"""
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h, w = frame.shape[:2]
            total_pixels = h * w
            
            # Detect brown/diseased areas
            brown_lower = np.array([10, 50, 20])
            brown_upper = np.array([25, 255, 200])
            brown_mask = cv2.inRange(hsv, brown_lower, brown_upper)
            brown_percentage = (cv2.countNonZero(brown_mask) / total_pixels) * 100
            
            # Detect yellow areas
            yellow_lower = np.array([20, 100, 100])
            yellow_upper = np.array([30, 255, 255])
            yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
            yellow_percentage = (cv2.countNonZero(yellow_mask) / total_pixels) * 100
            
            # Detect healthy green
            green_lower = np.array([35, 40, 40])
            green_upper = np.array([85, 255, 255])
            green_mask = cv2.inRange(hsv, green_lower, green_upper)
            green_percentage = (cv2.countNonZero(green_mask) / total_pixels) * 100
            
            return {
                'brown_spots': brown_percentage,
                'yellow_areas': yellow_percentage,
                'green_healthy': green_percentage
            }
        except:
            return {'brown_spots': 0, 'yellow_areas': 0, 'green_healthy': 0}
    
    def run_live_diagnosis(self, camera_index=0):
        """Run the fixed live diagnosis system"""
        print("🎥 Starting FIXED Live Leaf Diagnosis...")
        print("📝 Instructions:")
        print("   📷 Point camera at plant leaf")
        print("   🖱️ Press SPACEBAR to capture and analyze")
        print("   ❌ Press 'q' to quit")
        
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("❌ Could not open camera")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Draw interface
            display_frame = frame.copy()
            
            # Add semi-transparent overlay
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (10, 10), (1270, 120), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)
            
            # Add instructions
            cv2.putText(display_frame, f"FIXED Plant Disease Detection - {self.num_classes} classes", 
                       (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(display_frame, "Position leaf and press SPACEBAR to analyze", 
                       (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_frame, "Press 'q' to quit", 
                       (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('FIXED Live Plant Disease Detection', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("👋 Exiting...")
                break
            elif key == ord(' '):
                print("📸 Capturing and analyzing leaf...")
                result = self.process_capture(frame)
                if result:
                    print("✅ Analysis complete!")
        
        cap.release()
        cv2.destroyAllWindows()
    
    def process_capture(self, frame):
        """Process captured frame"""
        try:
            # Convert for AI model
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Get AI prediction
            prediction, confidence, all_probs = self.predict_disease(pil_image)
            
            # Visual analysis
            visual_features = self.analyze_leaf_visual(frame)
            
            # Save image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = f"{self.save_dir}/capture_{timestamp}.jpg"
            cv2.imwrite(image_path, frame)
            
            # Display results
            self.display_results(prediction, confidence, all_probs, visual_features, image_path)
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'all_probabilities': all_probs,
                'visual_features': visual_features,
                'image_path': image_path
            }
            
        except Exception as e:
            print(f"❌ Processing error: {e}")
            return None
    
    def display_results(self, prediction, confidence, all_probs, visual_features, image_path):
        """Display analysis results"""
        print("\n" + "=" * 70)
        print("🔬 LIVE LEAF ANALYSIS RESULTS")
        print("=" * 70)
        
        # Get disease info
        info = self.disease_info.get(prediction, {
            "status": f"🎯 {prediction}",
            "description": f"Detected: {prediction}",
            "actions": ["Monitor plant condition"]
        })
        
        print(f"\n{info['status']}")
        print(f"📈 Confidence: {confidence:.1%}")
        print(f"📝 {info['description']}")
        
        # Show visual analysis
        print(f"\n🔍 Visual Analysis:")
        print(f"   🟤 Brown spots: {visual_features['brown_spots']:.1f}%")
        print(f"   🟡 Yellow areas: {visual_features['yellow_areas']:.1f}%")
        print(f"   🟢 Green healthy: {visual_features['green_healthy']:.1f}%")
        
        # Show all class probabilities if multi-class
        if len(all_probs) > 2:
            print(f"\n📊 All Class Probabilities:")
            sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
            for class_name, prob in sorted_probs[:5]:  # Top 5
                print(f"   {'🎯' if class_name == prediction else '📊'} {class_name}: {prob:.1%}")
        
        # Show recommended actions
        actions = info.get('actions', [])
        if actions:
            print(f"\n💊 RECOMMENDED ACTIONS:")
            for action in actions[:3]:  # Top 3 actions
                print(f"   {action}")
        
        print(f"\n📁 Image saved: {image_path}")
        
        # Confidence warning
        if confidence < 0.7:
            print(f"\n⚠️ LOW CONFIDENCE WARNING:")
            print(f"   🔍 Try different angle or lighting")
            print(f"   📷 Ensure leaf is clearly visible")
            print(f"   🌿 Move closer to the leaf")
        
        print("=" * 70)

def main():
    """Main function"""
    import argparse
    parser = argparse.ArgumentParser(description="Fixed Live Plant Disease Detection")
    parser.add_argument("--model_path", default="my_leaf_model", help="Model directory")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    
    args = parser.parse_args()
    
    try:
        # Initialize system
        detector = FixedLiveDiagnosis(args.model_path)
        
        # Run live diagnosis
        detector.run_live_diagnosis(args.camera)
        
    except Exception as e:
        print(f"❌ System error: {e}")
        print("\n🔧 TROUBLESHOOTING:")
        print("1. Check if model exists: dir my_leaf_model")
        print("2. Try original model: python fixed_live_camera.py --model_path my_leaf_model")
        print("3. Check camera: python fixed_live_camera.py --camera 1")

if __name__ == "__main__":
    main()