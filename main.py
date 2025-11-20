import os
import whisper
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq


os.environ["XDG_CACHE_HOME"] = r"C:\whisper_cache"
os.environ["PATH"] += os.pathsep + r"C:\Users\gacha\Downloads\ffmpeg-2025-11-17-git-e94439e49b-essentials_build\ffmpeg-2025-11-17-git-e94439e49b-essentials_build\bin"

def audio_preprocessing():
    model=whisper.load_model("base")
    result=model.transcribe(r"c:\Users\gacha\OneDrive\Documentos\Sound Recordings\Recording (7).m4a",fp16=False)
    return result 

def getting_the_minutes_of_the_data(result):
    # prompt=f"""You are an AI assistant that extracts structured information from conversations
    # and generates a clean, professional meeting report.

    # Given the transcript below, perform ALL of the following:

    # 1. Extract a list of participants.
    # 2. Extract the main topics discussed.
    # 3. Extract decisions made.
    # 4. Extract any action items, including who is responsible and due dates.
    # 5. Generate a clear summary of the meeting.
    # 6. Format everything into a structured Meeting Minutes report with headers.

    # Return the final response in clean, human-readable report format.
    # Transcript:
    # {text}
    # """
    prompt=f"""You are supposed to Do an NER for this transcript and give me the NER present also I need a brief overview of what they are talking about 
            this is the transcript that is provided{result}"""

    groq_api_key=os.getenv("groq_api_key")

    llm=ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant",
                 temperature=0.1)
    response=llm.invoke([prompt])
    answer=response.content
    return answer

result=audio_preprocessing()
print(getting_the_minutes_of_the_data(result))


