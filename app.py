from __future__ import absolute_import, division, print_function
from flask import Flask, request
from timeit import default_timer as timer

import argparse
import os
import sys
import sox
import scipy.io.wavfile as wav
import pandas as pd
import numpy as np
from random import shuffle


from deepspeech.model import Model

app = Flask(__name__)

DS_HOME = "/home/krishna/Documents/DeepSpeech"

trimtfm = sox.Transformer()
trimtfm.vad()
trimtfm.vad(location=-1)
    
tempo_tfm1 = sox.Transformer()
tempo_tfm1.tempo(0.9)
	
tempo_tfm2 = sox.Transformer()
tempo_tfm2.tempo(1.1)   
	
speed_tfm1 = sox.Transformer()
speed_tfm1.speed(0.9)

speed_tfm2 = sox.Transformer()
speed_tfm2.speed(1.1)

# These constants control the beam search decoder

# Beam width used in the CTC decoder when building candidate transcriptions
BEAM_WIDTH = 500

# The alpha hyperparameter of the CTC decoder. Language Model weight
LM_WEIGHT = 1.75

# The beta hyperparameter of the CTC decoder. Word insertion weight (penalty)
WORD_COUNT_WEIGHT = 1.00

# Valid word insertion weight. This is used to lessen the word insertion penalty
# when the inserted word is part of the vocabulary
VALID_WORD_COUNT_WEIGHT = 1.00


# These constants are tied to the shape of the graph used (changing them changes
# the geometry of the first layer), so make sure you use the same constants that
# were used during training

# Number of MFCC features to use
N_FEATURES = 26

# Size of the context window used for producing timesteps in the input vector
N_CONTEXT = 9

@app.route("/")
def hello():
    return "Hello World!"

@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    audio = request.json['wav_url']
    
    model = "/home/krishna/Documents/DeepSpeech/models/digits/output_graph.pb"
    alphabet = "/home/krishna/Documents/DeepSpeech/pretrained/models/alphabet.txt"
    trie = "/home/krishna/Documents/DeepSpeech/pretrained/models/trie" 
    #audio =  "/home/krishna/Documents/Thesis_Data/custom/two.wav"
    
    #print('Loading model from file %s' % (model), file=sys.stderr)
    #model_load_start = timer()
    #ds = Model(model, N_FEATURES, N_CONTEXT, alphabet, BEAM_WIDTH)
    #model_load_end = timer() - model_load_start
    #print('Loaded model in %0.3fs.' % (model_load_end), file=sys.stderr)
    #
    #fs, audio = wav.read(audio)
    ## We can assume 16kHz
    #audio_length = len(audio) * ( 1 / 16000)
    #
    #print('Running inference.', file=sys.stderr)
    #inference_start = timer()
    #result = ds.stt(audio, fs)
    #inference_end = timer() - inference_start
    #print('Inference took %0.3fs for %0.3fs audio file.' % (inference_end, audio_length), file=sys.stderr)
    result = "test"
    return result
    
@app.route("/feedback", methods=["POST"])
def feedback():
    print(str(request.json))
    audio = request.json['wav_url']
    text  = request.json['wav_text']
    print("Arguments " + audio + " " + text)
    
    #fs, audio = wav.read(audio)  
    
    path, extension = audio.split(".") 
    trimmed = path + "_trimmed" + "." + extension
    trimtfm.build(audio, trimmed)
    
    augment_audio(trimmed, text)

    return "okay";   

def augment_audio(audio, text):
	path, extension = audio.split(".") 
	
	file1 = path + "_tfm1." + extension
	file2 = path + "_tfm2." + extension
	file3 = path + "_tfm3." + extension
	file4 = path + "_tfm4." + extension
	file5 = audio 
	
	files = [file1, file2, file3, file4, file5]
	shuffle(files)

	tempo_tfm1.build(audio, file1)
	tempo_tfm2.build(audio, file2)
	speed_tfm1.build(audio, file3) 
	speed_tfm2.build(audio, file4)	
	
	newtrain = createNewDataset("train", files[:4], text)
	newtest  = createNewDataset("test", files[4:], text)
	#createNewDataset("train", files[5:], text)
		
		
def createNewDataset(dataset_type, files, text):
	df = pd.read_csv(DS_HOME + "/data/digits/" + "digits-" + dataset_type + ".csv")
	classes = df.groupby("transcript").groups.keys()
	print(classes)
	df2 = pd.DataFrame(index=df.index.delete(slice(None)), columns=df.columns)
	for key in classes: 	 
		 some = df[df['transcript'] == key]
		 some = some.sample(len(files))
		 df2 = df2.append(some)
		 
	for somefile in files:
		df2.loc[len(df2)] = [somefile, os.path.getsize(somefile), text]	 
	
	#df2 = df2.drop(df.columns[[-1]], axis=1)
	df2.reset_index(drop=True, inplace=True)
	
	newfilename = "digits-" + dataset_type + "-" + text + ".csv"
	df2.to_csv(newfilename,  index=False)
	
	return newfilename
		
if __name__ == '__main__':
    app.run(debug=True)
