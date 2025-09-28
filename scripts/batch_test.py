import os
import glob
import time
import pandas as pd
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from model_integrator import ModelIntegrator

class LiveDataTester:
    def __init__(self, model_path=None):
        """Initialize live data testing system"""
        print("🧪 LIVE DATA TESTING SYSTEM")
        print("=" * 50)
        
        # Load model
        self.integrator = ModelIntegrator()
        if model_path:
            success = self.integrator.load_transformers_model(model_path)
        else:
            success = self.integrator.load_best_model()
        
        if not success:
            raise Exception("Could not load model for testing")
        
        self.processor = self.integrator.current_processor
        self.model = self.integrator.current_model
        self.model_info = self.integrator.get_model_info()
        
        # Results storage
        self.test_results = []
        self.results_dir = "live_test_results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        print(f"✅ Model loaded: {self.model_info['num_classes']} classes")
        
    def test_single_image(self, image_path, expected_result=None, show_image=False):
        """Test single image and return detailed results"""
        try:
            print(f"\n📊 Testing: {os.path.basename(image_path)}")
            
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt")
            
            # Get prediction
            start_time = time.time()
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            inference_time = time.time() - start_time
            
            # Extract results
            predicted_class_id = probabilities.argmax().item()
            confidence = probabilities[0][predicted_class_id].item()
            
            # Get class name
            id2label = self.model_info.get('id2label', {})
            predicted_label = id2label.get(str(predicted_class_id), f"class_{predicted_class_id}")
            
            # Get top 3 predictions
            top_3 = torch.topk(probabilities, min(3, len(id2label) if id2label else 2))
            alternatives = []
            for i in range(len(top_3.indices[0])):
                alt_id = top_3.indices[0][i].item()
                alt_conf = top_3.values[0][i].item()
                alt_name = id2label.get(str(alt_id), f"class_{alt_id}")
                alternatives.append((alt_name, alt_conf))
            
            # Analyze expected vs actual
            is_correct = None
            if expected_result:
                is_correct = expected_result.lower() in predicted_label.lower()
            
            # Create result object
            result = {
                'image_path': image_path,
                'image_name': os.path.basename(image_path),
                'predicted_label': predicted_label,
                'confidence': confidence,
                'inference_time': inference_time,
                'alternatives': alternatives,
                'expected': expected_result,
                'is_correct': is_correct,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Display results
            self.display_test_result(result)
            
            # Show image if requested
            if show_image:
                self.show_image_with_results(image_path, result)
            
            # Store result
            self.test_results.append(result)
            
            return result
            
        except Exception as e:
            print(f"❌ Error testing {image_path}: {e}")
            return None
    
    def display_test_result(self, result):
        """Display individual test result"""
        print(f"   🎯 Prediction: {result['predicted_label']}")
        print(f"   📈 Confidence: {result['confidence']:.1%}")
        print(f"   ⚡ Time: {result['inference_time']:.3f}s")
        
        if result['expected']:
            status = "✅ CORRECT" if result['is_correct'] else "❌ INCORRECT"
            print(f"   {status} (Expected: {result['expected']})")
        
        if len(result['alternatives']) > 1:
            print(f"   🔍 Alternatives:")
            for alt_name, alt_conf in result['alternatives'][1:3]:
                print(f"      📊 {alt_name}: {alt_conf:.1%}")
    
    def show_image_with_results(self, image_path, result):
        """Show image with prediction overlay"""
        # Load image with OpenCV
        img = cv2.imread(image_path)
        if img is None:
            return
        
        # Resize for display
        height, width = img.shape[:2]
        if width > 800:
            scale = 800 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            img = cv2.resize(img, (new_width, new_height))
        
        # Add prediction text
        overlay = img.copy()
        cv2.rectangle(overlay, (10, 10), (img.shape[1]-10, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        # Add text
        cv2.putText(img, f"Prediction: {result['predicted_label']}", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(img, f"Confidence: {result['confidence']:.1%}", 
                   (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, f"Time: {result['inference_time']:.3f}s", 
                   (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if result['expected']:
            status_color = (0, 255, 0) if result['is_correct'] else (0, 0, 255)
            status_text = "CORRECT" if result['is_correct'] else "INCORRECT"
            cv2.putText(img, f"Status: {status_text}", 
                       (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Show image
        cv2.imshow(f"Test Result: {result['image_name']}", img)
        cv2.waitKey(2000)  # Show for 2 seconds
        cv2.destroyAllWindows()
    
    def test_directory(self, directory_path, show_images=False, auto_detect_expected=True):
        """Test all images in a directory"""
        print(f"\n📁 Testing directory: {directory_path}")
        
        # Get all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(directory_path, ext)))
        
        if not image_files:
            print("❌ No image files found in directory")
            return
        
        print(f"📊 Found {len(image_files)} images to test")
        
        # Test each image
        for i, image_path in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}]", end=" ")
            
            # Auto-detect expected result from filename
            expected = None
            if auto_detect_expected:
                filename = os.path.basename(image_path).lower()
                if 'healthy' in filename:
                    expected = 'healthy'
                elif any(disease in filename for disease in ['blight', 'rust', 'scab', 'virus']):
                    expected = 'diseased'
            
            # Test image
            self.test_single_image(image_path, expected, show_images)
            
            # Small delay between tests
            time.sleep(0.5)
    
    def test_live_camera_import(self, duration_seconds=30):
        """Test with live camera data import"""
        print(f"\n📷 LIVE CAMERA TESTING ({duration_seconds} seconds)")
        print("Press SPACEBAR to capture and test, 'q' to quit early")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Could not open camera")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        start_time = time.time()
        capture_count = 0
        
        while time.time() - start_time < duration_seconds:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Draw instructions
            cv2.putText(frame, "Press SPACEBAR to test current frame", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Time remaining: {duration_seconds - int(time.time() - start_time)}s", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Captures: {capture_count}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Live Testing - Press SPACEBAR to capture', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                # Save frame and test it
                capture_count += 1
                temp_path = f"{self.results_dir}/live_capture_{capture_count}.jpg"
                cv2.imwrite(temp_path, frame)
                
                print(f"\n📸 Testing live capture {capture_count}...")
                result = self.test_single_image(temp_path, show_image=True)
                
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n✅ Live testing complete: {capture_count} captures tested")
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        if not self.test_results:
            print("❌ No test results to report")
            return
        
        print(f"\n📊 GENERATING TEST REPORT...")
        
        # Calculate statistics
        total_tests = len(self.test_results)
        correct_tests = sum(1 for r in self.test_results if r['is_correct'] == True)
        incorrect_tests = sum(1 for r in self.test_results if r['is_correct'] == False)
        unknown_tests = sum(1 for r in self.test_results if r['is_correct'] is None)
        
        avg_confidence = sum(r['confidence'] for r in self.test_results) / total_tests
        avg_time = sum(r['inference_time'] for r in self.test_results) / total_tests
        
        # Create DataFrame
        df = pd.DataFrame(self.test_results)
        
        # Generate report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"{self.results_dir}/test_report_{timestamp}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("🧪 LIVE DATA TESTING REPORT\n")
            f.write("=" * 60 + "\n")
            f.write(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"🤖 Model: {self.model_info['num_classes']} classes\n\n")
            
            f.write("📊 SUMMARY STATISTICS:\n")
            f.write(f"   🎯 Total Tests: {total_tests}\n")
            f.write(f"   ✅ Correct: {correct_tests}\n")
            f.write(f"   ❌ Incorrect: {incorrect_tests}\n")
            f.write(f"   ❓ Unknown: {unknown_tests}\n")
            
            if correct_tests + incorrect_tests > 0:
                accuracy = correct_tests / (correct_tests + incorrect_tests)
                f.write(f"   📈 Accuracy: {accuracy:.1%}\n")
            
            f.write(f"   🎯 Avg Confidence: {avg_confidence:.1%}\n")
            f.write(f"   ⚡ Avg Time: {avg_time:.3f}s\n\n")
            
            f.write("📋 DETAILED RESULTS:\n")
            for i, result in enumerate(self.test_results, 1):
                f.write(f"\n{i}. {result['image_name']}\n")
                f.write(f"   🎯 Predicted: {result['predicted_label']} ({result['confidence']:.1%})\n")
                f.write(f"   ⏱️ Time: {result['inference_time']:.3f}s\n")
                if result['expected']:
                    status = "✅ CORRECT" if result['is_correct'] else "❌ INCORRECT"
                    f.write(f"   📊 Status: {status} (Expected: {result['expected']})\n")
        
        # Save CSV for analysis
        csv_file = f"{self.results_dir}/test_results_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        
        print(f"✅ Report saved: {report_file}")
        print(f"📊 CSV data: {csv_file}")
        
        # Display summary
        self.display_test_summary()
        
        return report_file, csv_file
    
    def display_test_summary(self):
        """Display test summary in console"""
        if not self.test_results:
            return
        
        total_tests = len(self.test_results)
        correct_tests = sum(1 for r in self.test_results if r['is_correct'] == True)
        incorrect_tests = sum(1 for r in self.test_results if r['is_correct'] == False)
        
        avg_confidence = sum(r['confidence'] for r in self.test_results) / total_tests
        avg_time = sum(r['inference_time'] for r in self.test_results) / total_tests
        
        print(f"\n" + "=" * 60)
        print("🧪 LIVE DATA TESTING SUMMARY")
        print("=" * 60)
        print(f"📊 Total Tests: {total_tests}")
        print(f"✅ Correct: {correct_tests}")
        print(f"❌ Incorrect: {incorrect_tests}")
        
        if correct_tests + incorrect_tests > 0:
            accuracy = correct_tests / (correct_tests + incorrect_tests)
            print(f"📈 Accuracy: {accuracy:.1%}")
        
        print(f"🎯 Average Confidence: {avg_confidence:.1%}")
        print(f"⚡ Average Inference Time: {avg_time:.3f}s")
        print("=" * 60)

def main():
    """Main testing interface"""
    import argparse
    parser = argparse.ArgumentParser(description="Live data testing system")
    parser.add_argument("--model_path", help="Specific model to test")
    parser.add_argument("--test_dir", default="test", help="Directory to test")
    parser.add_argument("--show_images", action="store_true", help="Show images during testing")
    parser.add_argument("--live_camera", action="store_true", help="Test with live camera")
    parser.add_argument("--duration", type=int, default=30, help="Live camera test duration")
    
    args = parser.parse_args()
    
    try:
        # Initialize tester
        tester = LiveDataTester(args.model_path)
        
        if args.live_camera:
            # Live camera testing
            tester.test_live_camera_import(args.duration)
        else:
            # Directory testing
            tester.test_directory(args.test_dir, args.show_images)
        
        # Generate report
        tester.generate_test_report()
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")

if __name__ == "__main__":
    main()