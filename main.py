
# imports
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import soundfile as sf
from datasets import load_dataset
import torch
import pandas as pd
from itertools import permutations

# set engines (run every time)
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# load xvector containing speaker's voice characteristics from a dataset (run every time)
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

############ small test ############

# write text
inputs = processor(text="Hello, my dog is cute", return_tensors="pt")
speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
spectrogram = model.generate_speech(inputs["input_ids"], speaker_embeddings)

# save
sf.write("sounds/speech.wav", speech.numpy(), samplerate=16000)

############ medium test ############

# empty df
d = pd.DataFrame()
d.columns = ['s1', 's2', 's3']
d['s1'] = ['dee', 'doo', 'daa', 'tee', 'too', 'taa']
d['s2'] = ['pee', 'too', 'bee', 'boo', 'paa', 'daa']
d['s3'] = ['poo', 'too', 'too', 'pee', 'daa', 'dee']

# loop through df and make sounds
for i in range(len(d)):
    inputs = processor(text=f"{d['s1'][i]}" + f"{d['s2'][i]}" + f"{d['s3'][i]}", 
                       return_tensors="pt")
    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, 
                                   vocoder=vocoder)
    spectrogram = model.generate_speech(inputs["input_ids"], speaker_embeddings)
    sf.write("sounds/" + f"{d['s1'][i]}-" + f"{d['s2'][i]}-" + f"{d['s3'][i]}" + ".wav", 
             speech.numpy(), samplerate=16000)
    
############ fullscale ############

# list all possible syllables
syl2perm = ('pee', 'poo', 'paa', 'tee', 'too', 'taa', 'bee', 'boo', 'baa', 'dee', 'doo', 'daa')

# permute all combinations of three syllables
perm = permutations(syl2perm, r=3)
holding = []
for i in list(perm):
    holding.append(i)

holding = list(holding)

# create data frame of permuted syllables
d = pd.DataFrame(holding, columns=['s1', 's2', 's3'])
