import os
from flask import Flask, render_template
import asyncio
from aiohttp import web
from aiohttp_wsgi import WSGIHandler

# Importing Model Class
from models.model import *

# Initial App
app = Flask('aioflask')

# LSTM MODEL Instance
model_inst = LSTM_AUDIO()


# Global variables
data = b''
inx = 0
soln = ""


async def process_audio(fast_socket: web.WebSocketResponse):
    try:
        global soln
        await fast_socket.send_str(soln)
    except Exception as e:
        pass

# Socket --> For tracking


async def socket(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    # Global Vars
    global data
    global inx
    global soln

    while True:
        data += bytearray(await ws.receive_bytes())
        if (inx == 100):
            with open('./cache/output.wav', mode='bx') as f:
                f.write(data)
                soln = model_inst.load_audio_file()
                socket1 = await process_audio(ws)
                break
        inx += 1
        print(inx)


@app.route('/')
def home():
    #    execute_recording()
    return render_template('index.html')


# Main Function
if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    aio_app = web.Application()
    wsgi = WSGIHandler(app)
    aio_app.router.add_route('*', '/{path_info: *}', wsgi.handle_request)
    aio_app.router.add_route('GET', '/listen', socket)
    web.run_app(aio_app, port=5555)
