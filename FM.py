from faster_whisper import WhisperModel

model = WhisperModel("medium", device="cpu")
segments, info = model.transcribe("邹.m4a")
for seg in segments:
    print(seg.text)
