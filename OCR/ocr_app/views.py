from django.shortcuts import render
from PIL import Image
import io
import torch

from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

# --- Model ve İşlemciyi Global Olarak Yükle ---
try:
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "allenai/olmOCR-7B-0225-preview", torch_dtype=torch.bfloat16
    ).eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    print(f"Hata: OlmOCR modeli yüklenemedi. Hata detayı: {e}")


def index(request):
    ocr_text = ""

    if not MODEL_LOADED:
        ocr_text = "OCR motoru başlatılamadı. Lütfen sunucu loglarını kontrol edin."
        return render(request, "index.html", {"ocr_text": ocr_text})

    if request.method == "POST" and "image" in request.FILES:
        image_file = request.FILES["image"]

        try:
            image = Image.open(io.BytesIO(image_file.read())).convert("RGB")
            print(f"Görüntü: {image}")

            # **GELİŞMİŞ PROMPT OLUŞTURMA**
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract the text from the image."},
                        {"type": "image"},
                    ],
                }
            ]

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = processor(
                text=[text], images=[image], padding=True, return_tensors="pt"
            )
            inputs = {key: value.to(device) for (key, value) in inputs.items()}

            output = model.generate(**inputs, max_new_tokens=1024)

            generated_text = processor.batch_decode(output, skip_special_tokens=True)

            if generated_text:
                ocr_text = generated_text[0]
            else:
                ocr_text = "Metin çıkarılamadı."

        except Exception as e:
            ocr_text = f"Hata oluştu: {e}"

    return render(request, "index.html", {"ocr_text": ocr_text})
