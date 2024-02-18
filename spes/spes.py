import zmq
import os
import requests
import pyaudio
import audioop
import wave
from io import BytesIO
from pydub import AudioSegment
import logging
import sys
import glob

import threading
import time


import zmq

from tinyrpc import RPCClient
from tinyrpc.server import RPCServer
from tinyrpc.dispatch import RPCDispatcher
from tinyrpc.protocols.jsonrpc import JSONRPCProtocol
from tinyrpc.transports.zmq import ZmqClientTransport, ZmqServerTransport



import pybase64
from time import time, sleep
import numpy as np


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/spes.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ],
    encoding='utf-8'
)

IAM_TOKEN_ENVKEY = 'IAM_TOKEN'
SPES_ADDRESS_ENVKEY = 'SPES_ADDRESS'
HOPS_ADDRESS_ENVKEY = 'HOPS_ADDRESS'
HOPS_DEFAULT_ADDRESS = 'tcp://127.0.0.1:3002'
DEFAULT_ADDRESS = 'tcp://0.0.0.0:3001'
CHUNK = 1024 * 4
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
FOLDER_ID = "b1gplk6gop2viqk2q4b8"
AUDIO_ARTIFACTS_DIR = 'spes_artifacts'

''' Tuning parameters
'''
THRESHOLD = 80
PAUSE_TIME = 600

message_queue = []
dispatcher = RPCDispatcher()


def audio2ogg(audio_data: bytes, postfix: str = ''):
    buffered = BytesIO()
    wf = wave.open(buffered, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(audio_data)
    wf.close()
    # buffered.seek(0)
    # with open(f'{AUDIO_ARTIFACTS_DIR}/output{postfix}.wav', 'wb') as f:
    #     f.write(buffered.read())
    buffered.seek(0)
    audio = AudioSegment.from_wav(buffered)
    data = audio.export(format="ogg").read()
    with open(f'{AUDIO_ARTIFACTS_DIR}/output{postfix}.ogg', 'wb') as f:
        f.write(data)
    return data

def millis():
    return int(time() * 1000)

def ocr(audio_data, iam_token, folder_id):
    url = "https://stt.api.cloud.yandex.net/speech/v1/stt:recognize"
    headers = {
        "Authorization": f"Bearer {iam_token}",
    }
    params = {
        "lang": "ru",
        "folderId": folder_id,
        "format": "oggopus"
    }
    response = requests.post(url, params=params, data=audio_data, headers=headers)
    if response.status_code == 200:
        return response.json()['result']
    else:
        raise RuntimeError('ocr runtime error', response)

def get_voice_portion():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    pause_state = True
    pause_start_time = millis()
    recording = False
    frames = []
    counter = 0
    try:
        while True:
            data = stream.read(CHUNK)
            rms = audioop.rms(data, 2)

            if pause_state and not recording and rms > THRESHOLD:
                audio_start_t = millis()
                logging.info(f'start recording {counter} segment')
                recording = True

            if recording:
                frames.append(data)
            
            if rms < THRESHOLD:
                if not pause_state:
                    pause_state = True
                    pause_start_time = millis()
                if recording and pause_state and (millis() - pause_start_time) > PAUSE_TIME:
                    recording = False
                    logging.info(f'stop recording, voice length: {millis() - audio_start_t} ms')
                    audio_data = b''.join(frames)
                    yield audio_data
                    frames = []
                    counter += 1
    except KeyboardInterrupt:
        logging.info('closing audio stream')
        stream.stop_stream()
        stream.close()
        p.terminate()


def zmq_try_send_string(socket, msg):
    delay = 0.1
    max_attempts = 6
    '''
    
    '''
    for attempt in range(max_attempts):
        try:
            socket.send_string(msg, zmq.NOBLOCK)
            #delay = 0.1
            success = True
            break  # If send is successful, break the loop.
        except zmq.Again as e:
            logging.info(f'retry sending message, wait {delay}')
            sleep(delay)
            delay += delay

@dispatcher.public
def get_messages():
    global message_queue
    tmp = message_queue.copy()
    message_queue = []
    return tmp

@dispatcher.public
def put_messages(l):
    global message_queue
    message_queue.extend(l)

def create_mic_daemon():
    pass

def create_api_service_daemon(spes_server):
    def daemon_function():
        spes_server.serve_forever()
    daemon_thread = threading.Thread(target=daemon_function)
    daemon_thread.daemon = True
    daemon_thread.start()

def main(iam_token: str = '', spes_address: str = DEFAULT_ADDRESS, hops_address: str = HOPS_DEFAULT_ADDRESS):
    if IAM_TOKEN_ENVKEY in os.environ:
        iam_token = os.environ[IAM_TOKEN_ENVKEY]
    if SPES_ADDRESS_ENVKEY in os.environ:
        spes_address = os.environ[SPES_ADDRESS_ENVKEY]
    if HOPS_ADDRESS_ENVKEY in os.environ:
        hops_address = os.environ[HOPS_ADDRESS_ENVKEY]
    
    assert(iam_token)
    assert(spes_address)
    assert(hops_address)

    
    if os.path.exists(AUDIO_ARTIFACTS_DIR):
        lst = glob.glob(os.path.join(AUDIO_ARTIFACTS_DIR, '*'))
        for f in lst:
            os.remove(f)
    else:
        os.mkdir(AUDIO_ARTIFACTS_DIR)

    zmq_ctx = zmq.Context()
    
    hops_client = RPCClient(
        JSONRPCProtocol(),
        ZmqClientTransport.create(zmq_ctx, hops_address)
    )


    global spes_server
    spes_server = RPCServer(
        ZmqServerTransport.create(zmq_ctx, spes_address),
        JSONRPCProtocol(),
        dispatcher
    )

    hops_api_endpoint = hops_client.get_proxy()

    create_api_service_daemon(spes_server)

    it = get_voice_portion()
    counter = 0
    try:
        while True:
            voice = next(it)
            counter += 1
            data_int16 = np.frombuffer(voice, dtype=np.int16)
            data_float16 = (np.float32(data_int16) / np.iinfo(np.int16).max).astype(np.float16)
            voice_b64enc = pybase64.b64encode_as_string(data_float16.tobytes())
            simil = hops_api_endpoint.get_vocal_similarity(voice_b64enc)
            logging.info(f'voice detector {simil}')
            ogg_data = audio2ogg(voice, str(counter))
            if simil > 0.99:
                ocr_start_t = millis()
                txt = ocr(ogg_data, iam_token, FOLDER_ID)
                logging.info(f'{txt}, ocr took: {millis() - ocr_start_t} ms')
                if txt:
                    message_queue.append(txt)
                    
    except StopIteration:
        print(f'Ctrl-C: stopping service')
        zmq_ctx.destroy()
        exit(0)
    except KeyboardInterrupt:
        print(f'Ctrl-C: stopping service')
        zmq_ctx.destroy()
        exit(0)

if __name__ == "__main__":
    main()