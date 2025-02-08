import vosk
import pyaudio
import json
import os

# Here I have downloaded this model to my PC, extracted the files 
# and saved it in local directory
# Set the model path
model_path = "./vosk-model-fr-0.22"
if not os.path.exists(model_path):
    print("Downloading and unzipping speech recognition model....")
    os.system("wget https://alphacephei.com/vosk/models/vosk-model-fr-0.22.zip;unzip vosk-model-fr-0.22.zip")

# Initialize the model with model-path
model = vosk.Model(model_path)

# Open the microphone stream
p = pyaudio.PyAudio()

def do_recognize():
    # Create a recognizer
    rec = vosk.KaldiRecognizer(model, 16000)

    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    frames_per_buffer=8192)


    print("Listening for speech.")
    while True:
        data = stream.read(8192)#read in chunks of 4096 bytes
        if rec.AcceptWaveform(data):#accept waveform of input voice

            # print(recognized_text)        
            recognized_text = json.loads(rec.Result())['text']
            print(recognized_text)
            if len(recognized_text) > 0:
                stream.stop_stream()
                stream.close()
                return recognized_text
    
        #else:
        #    result = json.loads(rec.FinalResult())
        #    recognized_text = result['text']
        #    print(recognized_text)
        #    if "stop" in recognized_text:
        #        return recognized_text
            #print(recognized_text)
            
    # Parse the JSON result and get the recognized text
    #result = json.loads(rec.FinalResult())
    #recognized_text = result['text']
    #print(recognized_text)
                
# Stop and close the stream
#stream.stop_stream()
#stream.close()


#text = do_recognize()
#print(text)

# Terminate the PyAudio object
#p.terminate()
