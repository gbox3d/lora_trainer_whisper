# test_run_ct2.py
from faster_whisper import WhisperModel
import time

# 변환된 모델 경로
model_path = "outputs/ct2_small"
# 테스트할 오디오 파일 (경로 수정 필요)
audio_path = "datasets/Sample/wav/SPK014/SPK014KBSCU001/SPK014KBSCU001F001.wav" 

print(f"🚀 Loading CT2 Model from {model_path}...")
# device="cuda"로 설정하면 GPU 사용
model = WhisperModel(model_path, device="cuda", compute_type="float16")

print("🎤 Transcribing...")
start = time.time()

segments, info = model.transcribe(audio_path, language="ko", beam_size=5)

print(f"\n[Detected Language]: {info.language} (Probability: {info.language_probability:.2f})")
print("-" * 30)

full_text = ""
for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
    full_text += segment.text

print("-" * 30)
end = time.time()
print(f"✅ Total Time: {end - start:.4f} sec")