from flask import Flask, render_template, request,jsonify
from sam import Main
import logging
import random
import json
import torch
from brain import NeuralNet
from NeuralNetwork import bag_of_words, tokenize
from listen import listen
from speak import say
from task import NonInputExecution, InputExecution

#logging.basicConfig(level=logging.DEBUG,filename="app.log",format='%(asctime)s %(message)s',handlers=[logging.StreamHandler()])

#logger = logging.getLogger()

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/start', methods=['POST', 'GET'])
def start():
    if request.method == 'GET':
        try:
            while True:
                Main()
        except Exception as e:
            e
        return render_template('index.html')
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
