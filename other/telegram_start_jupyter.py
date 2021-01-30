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
import subprocess
import time
from os.path import join
import hashlib

token = '1391843927:AAEGeze6Pd2LbhtnZ3-__kTGN3lnurvaE0E'

def get_updates():
    r = requests.get('https://api.telegram.org/bot' + token + '/getupdates')
    return r


def start_jupyter():
    cmd_text = 'cd ~/myjupyter/'
    cmd_text += ' & /opt/slurm/bin/sbatch ~/myjupyter/run-jupyter.sbatch'
    r = subprocess.run(cmd_text, shell=True, capture_output=True)
    return r


def get_output_file(job_id: int, chat_id: str):
    file_name = ''.join(('slurm-', str(job_id), '.out'))
    file_path = join('/home/ag7531', file_name)
    n = 0
    while True:
        n += 1
        if n >= 6:
            return 'Output file not found...'
        try:
            send_message('Looking for output file ' + file_path, chat_id)
            with open(file_path) as f:
                send_message('Found the file!', chat_id)
                lines = f.readlines()
                for line in lines:
                    if '127.0.0.1' in line:
                        return lines
        except FileNotFoundError:
            pass
        time.sleep(20)


def check_user(user_id: int):
    try:
        with open('.telegram_users') as f:
            return user_id in [int(s.strip()) for s in f.readlines()]
    except FileNotFoundError:
        return False


def register_new_user(user_id):
    with open('.telegram_users', 'a') as f:
        f.write(str(user_id) + '\n')


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
        user_id = int(update['message']['from']['id'])
        chat_id = update['message']['chat']['id']
        if check_user(user_id):
            if update['message']['text'] == 'start jupyter':
                send_message('Trying to start jupyter...', chat_id)
                r = start_jupyter()
                s = r.stdout.decode()
                send_message(s, chat_id)
                output_file = get_output_file(int(s.split()[-1]), chat_id)
                print(f'output file content: {output_file}')
                for line in output_file:
                    send_message(line, chat_id)
                send_message('Done!', chat_id)
            else:
                send_message('Did not understand your request, sorry.', chat_id)
        else:
            message = update['message']['text']
            hash_m = hashlib.sha256(message.encode()).hexdigest()
            if  hash_m == '141398e3d78065d224cc535a984d7aa000a0429b1ead2687f16a81e05c8f5f41':
                register_new_user(user_id)
                send_message('Thanks, you are now registered.', chat_id)
            else:
                send_message('You are not registered as a user yet.', chat_id)
                send_message('Please reply with the password', chat_id)
