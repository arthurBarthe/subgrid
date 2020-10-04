#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 02:51:59 2020

@author: arthur
In this script we read the updates for a telegram bot and let the user
start a program by text.
"""

import requests
import json
import sys
from telegram import send_message

token = '1391843927:AAEGeze6Pd2LbhtnZ3-__kTGN3lnurvaE0E'
chat_id = '1330475894'


def get_updates(id_: int = None):
    r = requests.get('https://api.telegram.org/bot' + token + '/getupdates')
    return r


# We read updates, if we find the expected message we start the jupyter script
r = get_updates()
updates = json.loads(r.text)

if not updates['ok']:
    sys.exit(1)

# Read the last update from file
try:
    with open('.last_update_id', 'r') as f:
        last_update_id = f.readline()
        if last_update_id == '':
            last_update_id = 0
        else:
            last_update_id = int(last_update_id)
except FileNotFoundError:
    last_update_id = 0

updates = updates['result']
for update in updates:
    update_id = update['update_id']
    if update_id > last_update_id:
        # Update last_update_id in file
        with open('.last_update_id', 'w') as f:
            f.write(str(update_id))
        # Check user
        if update['message']['from']['id'] == 1330475894:
            if update['message']['text'] == 'start jupyter':
                send_message('Starting jupyter for you!')
            else:
                send_message('Did not understand')
        else:
            send_message('Unauthorized user')
