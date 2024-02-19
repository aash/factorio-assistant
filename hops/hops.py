import os
from collections.abc import Callable, Iterable, Mapping
from typing import Any
import zmq

from tinyrpc.server import RPCServer
from tinyrpc.dispatch import RPCDispatcher
from tinyrpc.protocols.jsonrpc import JSONRPCProtocol
from tinyrpc.transports.zmq import ZmqServerTransport

import cv2 as cv
import numpy as np

import spacy
import pymorphy2
from ultralytics import YOLO

import scipy.signal as sg
from spleeter.separator import Separator


MODEL_PATH = 'best.pt'
HOPS_DEFAULT_ADDRESS = 'tcp://0.0.0.0:3002'

ctx = zmq.Context()
dispatcher = RPCDispatcher()
transport = ZmqServerTransport.create(ctx, HOPS_DEFAULT_ADDRESS)

nlp = spacy.load("ru_core_news_lg")
morph = pymorphy2.MorphAnalyzer()
assert(os.path.exists(MODEL_PATH))
yolo = YOLO(MODEL_PATH)

global separator
separator = Separator('spleeter:2stems', multiprocess=False)

global rpc_server
rpc_server = RPCServer(
    transport,
    JSONRPCProtocol(),
    dispatcher
)

@dispatcher.public
def translate_string(msg):
    doc = nlp(msg)
    verb, noun, num, adj = [None]*4
    for token in doc:
        if token.dep_ == "punct":
            continue
        if token.dep_ == 'ROOT' and token.pos_ == 'VERB':
            verb = morph.parse(token.text)[0].normal_form
        if token.pos_ == 'NUM':
            num = morph.parse(token.text)[0].normal_form
        if token.pos_ == 'NOUN':
            noun = morph.parse(token.text)[0].normal_form
            for t in token.children:
                if t.pos_ == 'ADJ':
                    adj = morph.parse(t.text)[0].normal_form
        print(f"Токен: {token.text}, Часть речи: {token.pos_}, Родитель: {token.head.text}, Связь: {token.dep_}")

    ru_key = noun
    if adj:
        ru_key = f'{adj} {ru_key}'

    print(verb, noun, adj, num, repr(ru_key))

    match verb:
        case 'выйти':
            return {'command': 'quit'}
        case 'сделать':
            return {'command': 'make', 'what': ru_key, 'num': int(num)}
from time import time

def millis():
    return time() * 1000

import pybase64
@dispatcher.public
def parse_screenshot(im_b64):
    im_enc = np.asarray(bytearray(pybase64.b64decode(im_b64)), dtype="uint8")
    im = cv.imdecode(im_enc, cv.IMREAD_COLOR)
    im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    results = yolo.predict(im, stream=True, verbose=False, conf=0.5)
    #print(millis() - t)
    out = []
    for result in results:
        for b in result.boxes:
            rect = list(map(int, b.xywh.tolist()[0]))
            rect[0] -= rect[2]//2
            rect[1] -= rect[3]//2
            out.append(rect)
    return out

@dispatcher.public
def get_vocal_similarity(audio_b64):
    separator = globals()['separator']
    x = np.frombuffer(pybase64.b64decode_as_bytearray(audio_b64), dtype="float16")
    x = np.array(x,ndmin = 2)
    x = np.transpose(x)
    res = separator.separate(x)
    x = np.transpose(x)[0]
    x1 = res['vocals'][:,0]
    norm_cross_corr = sg.correlate(x, x1, mode='valid')
    norm = np.sqrt(np.sum(x ** 2) * np.sum(x1 ** 2))
    if norm != 0:
        norm_cross_corr /= norm
    similarity = np.max(np.abs(norm_cross_corr))
    return float(similarity)

rpc_server.serve_forever()

