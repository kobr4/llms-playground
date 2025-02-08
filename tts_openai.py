from pathlib import Path
from openai import OpenAI
# Import pygame for playing the converted audio
import pygame

client = OpenAI()


def do_tts(text,language):
    speech_file_path = Path(__file__).parent / "speech.mp3"
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text,
    )
    response.stream_to_file(speech_file_path)

    # Initialize the mixer module
    pygame.mixer.init()

    # Load the mp3 file
    pygame.mixer.music.load("speech.mp3")

    # Play the loaded mp3 file
    pygame.mixer.music.play()
