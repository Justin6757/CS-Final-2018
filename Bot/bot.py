#!/usr/bin/env python
# -*- coding: utf-8 -*-

import asyncio
import pickle
import time

import discord
from discord import Game
from discord.ext.commands import Bot

from lstm_network import create

NEURAL_NET = create()


def get_sentiment(sentence):
    prediction = NEURAL_NET.predict(sentence)
    negative_score = prediction[0]
    non_negative_score = prediction[1]
    print(f'Positive: {non_negative_score}\nNegative: {negative_score}\nComposite: {non_negative_score-negative_score}')
    return non_negative_score - negative_score


# class for member info
class DiscordMember:
    def __init__(self, uid, time_):
        self.id = uid
        self.score = 25
        self.time_ = time_

    def __eq__(self, other):
        if self.id == other.id:
            return True
        return False

    def __str__(self):
        print("ID: " + self.id + "\n")
        print("Score: " + score + "\n")
        print("Time Since Last Message: " + self.time_ + "\n")
        print("--------------------------------")


# loads data from previous session of bot
try:
    memberList = pickle.load(open("user.pickle", "rb"))
except (OSError, IOError) as e:
    memberList = []
    pickle.dump(memberList, open("user.pickle", "wb"), protocol=pickle.HIGHEST_PROTOCOL)

# Example load and dump using pickle:
# a = {'hello': 'world'}
# pickle.dump(a, open('filename.pickle', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
#
# b = pickle.load(open('filename.pickle', 'rb')

BOT_PREFIX = '!'
# Get at https://discordapp.com/developers/applications/me
TOKEN = 'NDQzNDQzNzUyNzI4NjU3OTIx.DdNdWQ.3HElcDFaWvTdVyn18XlrTZGhZpM'

client = Bot(command_prefix=BOT_PREFIX)

MAX_SCORE = 25
WARNING_SCORE = 15
BAN_SCORE = 10


@client.event
async def on_ready():
    await client.change_presence(game=Game(name='positively'))
    print('Logged in as ' + client.user.name)
    servers = list(client.servers)
    for server in servers:
        for member in server.members:
            temp = DiscordMember(member.id, time.time())
            if temp not in memberList:
                memberList.append(temp)
    for member in memberList:
        print(member)


async def list_servers():
    await client.wait_until_ready()
    while not client.is_closed:
        print('Current servers:')
        for server in client.servers:
            print(server.name)
        await asyncio.sleep(600)


@client.event
async def on_message(message):
    await client.process_commands(message)
    if message.content != '!score' and message.author.id != client.user.id:
        try:
            score_change = 0
            # for sentence in re.split(r'\. |\? |! ', message.content):
            #     if sentence:
            #         score_change += min(analyze(sentence)[1].get('watson'), 0)
            # message_toxicity_string, toxicity_dict = analyze(message.content)
            # await client.send_message(message.channel, message_toxicity_string)
        except TypeError:  # returned none
            print('No message to analyze')
            return

        # Update score

        current_time = time.time()
        old_time = current_time

        temp = DiscordMember(message.author.id, time.time())
        if temp not in memberList:
            memberList.append(temp)

        for user in memberList:
            if user.id == message.author.id:
                temp = user
                prev_score = user.score
                old_time = user.time

        time_points = (current_time - old_time) / 600

        new_score = min(prev_score + time_points, MAX_SCORE) + score_change

        temp.score = new_score
        temp.time = current_time

        if new_score <= BAN_SCORE:
            try:
                await client.ban(message.server.get_member(message.author.id), delete_message_days=0)
            except discord.errors.Forbidden:
                print('Privilege too low')
            else:
                memberList.remove(temp)

        elif new_score <= WARNING_SCORE:
            await client.send_message(message.channel,
                                      f'**WARNING, <@{message.author.id}>, your positivity score is very low '
                                      f'({"{0:0.1f}".format(new_score)}/{MAX_SCORE})**'
                                      f'\nYou will be banned if your score reaches {BAN_SCORE} or below.')


@client.command(pass_context=True)
async def score(ctx):
    temp = DiscordMember(ctx.message.author.id, time.time())
    if temp not in memberList:
        memberList.append(temp)

    current_time = time.time()
    old_time = current_time

    for user in memberList:
        if user.id == ctx.message.author.id:
            temp = user
            prev_score = user.score
            old_time = user.time

    time_points = (current_time - old_time) / 600

    temp.score = min(prev_score + time_points, MAX_SCORE)
    temp.time = current_time

    await client.send_message(ctx.message.channel,
                              f'{ctx.message.author}\'s score is '
                              f'{"{0:0.1f}".format(min(prev_score + time_points, MAX_SCORE))}/{MAX_SCORE}')


client.loop.create_task(list_servers())
client.run(TOKEN)
