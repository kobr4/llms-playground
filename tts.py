# Import the required module for text 
# to speech conversion
from gtts import gTTS

# Import pygame for playing the converted audio
import pygame

import time

def do_tts(text,language):

    # Passing the text and language to the engine, 
    # here we have marked slow=False. Which tells 
    # the module that the converted audio should 
    # have a high speed
    myobj = gTTS(text=text, lang=language, slow=False)

    # Saving the converted audio in a mp3 file named
    # welcome 
    myobj.save("speech.mp3")

    # Initialize the mixer module
    pygame.mixer.init()

    # Load the mp3 file
    pygame.mixer.music.load("speech.mp3")

    # Play the loaded mp3 file
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        time.sleep(1)
