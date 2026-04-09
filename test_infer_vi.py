import os
import sys
import torch
import numpy as np
import soundfile as sf

# Thêm đường dẫn vào sys.path để import từ StyleTTS2-lite-infer (Ưu tiên hàng đầu)
sys.path.insert(0, os.path.join(os.getcwd(), 'StyleTTS2-lite-infer'))

from inference import StyleTTS2

def main():
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Paths
    config_path = "StyleTTS2-lite-infer/Models/config.yaml"
    model_path = "StyleTTS2-lite-infer/Models/base_model_120k_vi.pth"
    
    # Load model
    print("Loading model (1.69GB)...")
    stts2 = StyleTTS2(config_path, model_path).eval().to(device)
    
    # Reference audio
    ref_dir = "StyleTTS2-lite-infer/reference_audio"
    ref_files = [f for f in os.listdir(ref_dir) if f.endswith('.wav')]
    if not ref_files:
        print("Error: No reference audio files found!")
        return
    
    ref_path = os.path.join(ref_dir, ref_files[0])
    print(f"Using reference audio: {ref_path}")
    
    # Test content
    text = "Chào bạn, tôi là trợ lý AI. Tôi đang kiểm tra chất lượng giọng đọc tiếng Việt của mô hình Style T T S 2 lite."
    print(f"Synthesizing text: {text}")
    
    # Get styles
    speakers = {
        "id_1": {
            "path": ref_path,
            "lang": "vi",
            "speed": 1.0
        }
    }
    
    with torch.no_grad():
        styles = stts2.get_styles(speakers, denoise=0.6, avg_style=True)
        # Generate
        audio = stts2.generate(text, styles, stabilize=True, n_merge=18, default_speaker="[id_1]")
        
    # Save output
    output_path = "test_output_vi.wav"
    sf.write(output_path, audio, 24000)
    print(f"Success! Output saved at: {output_path}")

if __name__ == "__main__":
    main()
