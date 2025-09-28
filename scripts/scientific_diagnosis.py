import argparse
import os
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import cv2
import numpy as np
import time
from datetime import datetime


def load_image(image_path: str) -> Image.Image:
    return Image.open(image_path).convert("RGB")


class EnhancedDiseaseAnalyzer:
    def __init__(self, model_path="./my_leaf_model"):
        """Initialize the enhanced disease analyzer"""
        self.processor = AutoImageProcessor.from_pretrained(model_path)
        self.model = AutoModelForImageClassification.from_pretrained(model_path)
        self.model.eval()
        
        # Create directory for analysis results
        self.save_dir = "analysis_results"
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Disease information with detailed solutions (from live_camera_diagnosis)
        self.disease_info = {
            "diseased": {
                "status": "🔴 DISEASE DETECTED",
                "description": "The leaf shows signs of disease that require immediate attention",
                "immediate_actions": [
                    "🚨 Isolate the plant to prevent spread to other plants",
                    "✂️ Remove affected leaves using sterilized pruning shears",
                    "🌬️ Improve air circulation around the plant",
                    "💧 Avoid watering leaves directly - water at soil level",
                    "🧹 Clean up fallen debris around the plant"
                ],
                "treatments": [
                    "🧴 Apply appropriate fungicide or bactericide as per disease type",
                    "⏰ Adjust watering schedule - water in morning, not evening",
                    "🌿 Consider organic treatments like neem oil or copper fungicide",
                    "📊 Monitor plant daily and track disease progression",
                    "🏥 Consult agricultural expert for severe infections"
                ],
                "prevention": [
                    "🌱 Use disease-resistant plant varieties in future",
                    "📏 Maintain proper spacing between plants",
                    "🔄 Practice crop rotation if applicable",
                    "🧽 Keep gardening tools clean and sterilized"
                ]
            },
            "healthy": {
                "status": "🟢 HEALTHY PLANT",
                "description": "The leaf appears healthy with no visible disease symptoms",
                "maintenance": [
                    "✅ Continue current care routine - it's working well",
                    "👀 Monitor plant weekly for early disease detection",
                    "🌱 Maintain proper nutrition with balanced fertilizer",
                    "💧 Keep consistent watering schedule without overwatering",
                    "☀️ Ensure adequate sunlight and air circulation"
                ],
                "prevention": [
                    "🔍 Regular inspection schedule (weekly checks)",
                    "🌬️ Good air circulation around plants",
                    "📏 Proper spacing between plants",
                    "🧹 Clean gardening tools between uses",
                    "🍂 Remove fallen leaves and debris regularly"
                ]
            }
        }

    def predict_disease(self, image):
        """Predict disease from PIL image"""
        inputs = self.processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
        predicted_class_id = probabilities.argmax().item()
        confidence = probabilities[0][predicted_class_id].item()
        
        # Get label name
        id2label = self.model.config.id2label if hasattr(self.model.config, "id2label") else {0: "diseased", 1: "healthy"}
        predicted_label = id2label.get(predicted_class_id, "unknown")
        
        return predicted_label, confidence

    def analyze_leaf_features(self, image_path):
        """Enhanced visual analysis for disease classification (from live_camera_diagnosis)"""
        try:
            # Load image with OpenCV
            frame = cv2.imread(image_path)
            if frame is None:
                return {"error": "Could not load image"}
            
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h, w = frame.shape[:2]
            total_pixels = h * w
            
            # Detect brown/yellow spots (disease indicators)
            brown_lower = np.array([10, 50, 20])
            brown_upper = np.array([25, 255, 200])
            brown_mask = cv2.inRange(hsv, brown_lower, brown_upper)
            brown_percentage = (cv2.countNonZero(brown_mask) / total_pixels) * 100
            
            # Detect yellow areas (stress/disease indicators)
            yellow_lower = np.array([20, 100, 100])
            yellow_upper = np.array([30, 255, 255])
            yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
            yellow_percentage = (cv2.countNonZero(yellow_mask) / total_pixels) * 100
            
            # Detect healthy green areas
            green_lower = np.array([35, 40, 40])
            green_upper = np.array([85, 255, 255])
            green_mask = cv2.inRange(hsv, green_lower, green_upper)
            green_percentage = (cv2.countNonZero(green_mask) / total_pixels) * 100
            
            return {
                'brown_spots': brown_percentage,
                'yellow_areas': yellow_percentage,
                'green_healthy': green_percentage
            }
            
        except Exception as e:
            return {"error": f"Visual analysis failed: {e}"}

    def show_analysis_animation(self, duration=2.0):
        """Show AI analysis animation for specified duration"""
        start_time = time.time()
        analysis_steps = [
            "🔬 Initializing AI analysis...",
            "📊 Processing image features...",
            "🧠 Running neural network inference...",
            "🔍 Analyzing leaf patterns...",
            "⚗️ Detecting disease indicators...",
            "📈 Computing confidence scores...",
            "✅ Analysis complete!"
        ]
        
        step_duration = duration / len(analysis_steps)
        
        for i, step in enumerate(analysis_steps):
            elapsed = time.time() - start_time
            if elapsed >= duration:
                break
                
            # Create animated dots
            dots = "." * ((i % 3) + 1)
            print(f"\r{step}{dots}   ", end="", flush=True)
            time.sleep(step_duration)
        
        print()  # New line after animation

    def generate_diagnosis_report(self, image_path, prediction, confidence, features, timestamp):
        """Generate detailed diagnosis report"""
        report_filename = f"{self.save_dir}/report_{timestamp}.txt"
        info = self.disease_info.get(prediction, {})
        
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write("🔬 PLANT DISEASE DIAGNOSIS REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"📅 Date & Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"📷 Image File: {image_path}\n")
            f.write(f"🤖 AI Model: Fine-tuned Plant Disease Classifier\n\n")
            
            f.write("📊 DIAGNOSIS RESULTS:\n")
            f.write(f"   🎯 Classification: {prediction.upper()}\n")
            f.write(f"   📈 Confidence Level: {confidence:.1%}\n")
            f.write(f"   {info.get('status', '')}\n\n")
            
            f.write("📝 DESCRIPTION:\n")
            f.write(f"   {info.get('description', 'No description available')}\n\n")
            
            if "error" not in features:
                f.write("🔍 VISUAL ANALYSIS:\n")
                f.write(f"   🟤 Brown/Disease spots: {features['brown_spots']:.1f}%\n")
                f.write(f"   🟡 Yellow stress areas: {features['yellow_areas']:.1f}%\n")
                f.write(f"   🟢 Healthy green areas: {features['green_healthy']:.1f}%\n\n")
            
            # Add confidence assessment
            f.write("📊 CONFIDENCE ASSESSMENT:\n")
            if confidence > 0.8:
                f.write("   🎯 HIGH CONFIDENCE - Diagnosis is very reliable\n")
            elif confidence > 0.6:
                f.write("   ⚖️ MODERATE CONFIDENCE - Results are fairly reliable\n")
            else:
                f.write("   ⚠️ LOW CONFIDENCE - Consider expert consultation\n")
            f.write("\n")
            
            if prediction == "diseased":
                f.write("🚨 IMMEDIATE ACTION REQUIRED:\n")
                for action in info.get('immediate_actions', []):
                    f.write(f"   {action}\n")
                f.write("\n")
                
                f.write("💊 TREATMENT RECOMMENDATIONS:\n")
                for treatment in info.get('treatments', []):
                    f.write(f"   {treatment}\n")
                f.write("\n")
                
                f.write("🛡️ PREVENTION MEASURES:\n")
                for prevention in info.get('prevention', []):
                    f.write(f"   {prevention}\n")
            else:
                f.write("🌱 MAINTENANCE RECOMMENDATIONS:\n")
                for maintenance in info.get('maintenance', []):
                    f.write(f"   {maintenance}\n")
                f.write("\n")
                
                f.write("🛡️ PREVENTION TIPS:\n")
                for prevention in info.get('prevention', []):
                    f.write(f"   {prevention}\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("📋 Report generated by AI Plant Disease Detection System\n")
            f.write("🔬 For questions, consult with agricultural experts\n")
        
        return report_filename

    def display_diagnosis_results(self, prediction, confidence, features, image_path, report_file):
        """Display diagnosis results in console (from live_camera_diagnosis)"""
        print("\n" + "=" * 60)
        print("🔬 DIAGNOSIS COMPLETE!")
        print("=" * 60)
        
        info = self.disease_info.get(prediction, {})
        print(f"\n{info.get('status', '')}")
        print(f"📈 Confidence: {confidence:.1%}")
        print(f"\n📝 {info.get('description', '')}")
        
        if "error" not in features:
            print(f"\n🔍 Visual Analysis:")
            print(f"   🟤 Brown spots: {features['brown_spots']:.1f}%")
            print(f"   🟡 Yellow areas: {features['yellow_areas']:.1f}%") 
            print(f"   🟢 Healthy green: {features['green_healthy']:.1f}%")
        
        if prediction == "diseased":
            print(f"\n🚨 IMMEDIATE ACTIONS:")
            for action in info.get('immediate_actions', [])[:3]:  # Show first 3
                print(f"   {action}")
            
            print(f"\n💊 KEY TREATMENTS:")
            for treatment in info.get('treatments', [])[:3]:  # Show first 3
                print(f"   {treatment}")
        else:
            print(f"\n🌱 MAINTENANCE TIPS:")
            for maintenance in info.get('maintenance', [])[:3]:  # Show first 3
                print(f"   {maintenance}")
        
        print(f"\n📁 Files saved:")
        print(f"   📷 Original image: {image_path}")
        print(f"   📄 Analysis report: {report_file}")
        
        print(f"\n💡 Tip: Check the detailed report file for complete recommendations!")
        print("=" * 60)

    def analyze_image(self, image_path):
        """Complete image analysis with all features"""
        print("\n" + "="*50)
        print("🤖 AI DIAGNOSIS IN PROGRESS")
        print("="*50)
        
        try:
            # Load image
            image = load_image(image_path)
            
            # Show analysis animation for 2 seconds
            self.show_analysis_animation(duration=2.0)
            
            # Get AI prediction
            prediction, confidence = self.predict_disease(image)
            
            # Perform visual analysis
            features = self.analyze_leaf_features(image_path)
            
            # Generate timestamp and report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = self.generate_diagnosis_report(
                image_path, prediction, confidence, features, timestamp
            )
            
            # Small pause before showing results
            print("🎯 Finalizing diagnosis...")
            time.sleep(0.5)
            
            # Display results
            self.display_diagnosis_results(prediction, confidence, features, image_path, report_filename)
            
            return {
                'prediction': prediction,
                'confidence': confidence, 
                'features': features,
                'image_file': image_path,
                'report_file': report_filename
            }
            
        except Exception as e:
            print(f"❌ Error during diagnosis: {e}")
            return None


def main():
    parser = argparse.ArgumentParser(description="Enhanced Scientific Plant Disease Analysis")
    parser.add_argument("image", type=str, help="Path to the input image")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./my_leaf_model",
        help="Directory containing the fine-tuned model",
    )
    parser.add_argument("--save_report", action="store_true", help="Save detailed report to file")
    
    args = parser.parse_args()

    # Check if model and image exist
    if not os.path.exists(args.model_dir):
        print(f"❌ Error: Model not found at {args.model_dir}")
        print("💡 Please train the model first using: python finetune_model.py")
        return
    
    if not os.path.exists(args.image):
        print(f"❌ Error: Image not found at {args.image}")
        return
    
    print("🔬 AI PLANT DISEASE DIAGNOSIS SYSTEM")
    print("=" * 50)
    print("📊 Enhanced Scientific Analysis Mode")
    print("=" * 50)
    print(f"🔍 Analyzing: {os.path.basename(args.image)}")
    
    # Initialize enhanced analyzer
    analyzer = EnhancedDiseaseAnalyzer(args.model_dir)
    
    # Run complete analysis
    result = analyzer.analyze_image(args.image)
    
    if result:
        print(f"\n✅ Analysis completed successfully!")
        print(f"🎯 Final Result: {result['prediction']} ({result['confidence']:.1%} confidence)")
    else:
        print(f"\n❌ Analysis failed")


if __name__ == "__main__":
    main()