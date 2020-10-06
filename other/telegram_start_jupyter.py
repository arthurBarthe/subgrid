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

token = '1391843927:AAEGeze6Pd2LbhtnZ3-__kTGN3lnurvaE0E'
chat_id = '1330475894'


def get_updates():
    r = requests.get('https://api.telegram.org/bot' + token + '/getupdates')
    return r


def start_jupyter():
    cmd_text = 'cd ~/myjupyter/'
    cmd_text += ' & /opt/slurm/bin/sbatch ~/myjupyter/run-jupyter.sbatch'
    r = subprocess.run(cmd_text, shell=True, capture_output=True)
    return r


def get_output_file(job_id: int):
    file_name = ''.join(('slurm-', str(job_id), '.out'))
    file_path = join('/home/ag7531', file_name)
    n = 0
    while True:
        n += 1
        time.sleep(20)
        try:
            send_message('Looking for output file ' + file_path)
            with open(file_path) as f:
                send_message('Found the file!')
                return f.readlines()
        except FileNotFoundError:
            if n >= 6:
                return 'Output file not found...'


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
                send_message('Trying to start jupyter...')
                r = start_jupyter()
                s = r.stdout.decode()
                send_message(s)
                output_file = get_output_file(int(s.split()[-1]))
                print(f'{output_file=}')
                send_message(output_file)
                send_message('Done!')
            else:
                send_message('Did not understand')
        else:
            send_message('Unauthorized user')
