#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 03:01:49 2020

@author: arthur
"""

import requests

bot_token =  1391843927:AAEGeze6Pd2LbhtnZ3-__kTGN3lnurvaE0E
chat_id = 1330475894


def send_message(text: str):
    parameters = {'chat_id': 1330475894, 'text': text}
    r = requests.get('https://api.telegram.org/bot' + token + '/sendMessage',
                     params=parameters)