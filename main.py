import os
import whisper
os.environ["XDG_CACHE_HOME"] = r"C:\whisper_cache"
os.environ["PATH"] += os.pathsep + r"C:\Users\gacha\Downloads\ffmpeg-2025-11-17-git-e94439e49b-essentials_build\ffmpeg-2025-11-17-git-e94439e49b-essentials_build\bin"

def audio_preprocessing():
    model=whisper.load_model("base")
    result=model.transcribe(r"c:\Users\gacha\OneDrive\Documentos\Sound Recordings\Recording (7).m4a",fp16=False)
    return result 


