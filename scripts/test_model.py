import argparse
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification


def load_image(image_path: str) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def main():
    parser = argparse.ArgumentParser(description="Test a fine-tuned plant leaf classifier.")
    parser.add_argument("image", type=str, help="Path to the input image")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./my_leaf_model",
        help="Directory containing the fine-tuned model and processor",
    )
    args = parser.parse_args()

    processor = AutoImageProcessor.from_pretrained(args.model_dir)
    model = AutoModelForImageClassification.from_pretrained(args.model_dir)

    image = load_image(args.image)
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred_id = int(torch.argmax(logits, dim=-1).item())

    id2label = model.config.id2label if hasattr(model.config, "id2label") else {}
    pred_label = id2label.get(pred_id, str(pred_id))

    print(f"predicted_class_id: {pred_id}")
    print(f"predicted_label: {pred_label}")


if __name__ == "__main__":
    main()


