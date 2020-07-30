import discord
import asyncio
import aiohttp
import aiofiles
import pickle
from single_feature_extraction import *
#from single_testing import *



TOKEN_bot = 'NTMyMjc1OTEwODM4MDU5MDEw.Dx2MOA.tpjN9cjPR5F1JNS3TCEBrfXDEr8'
TOKEN = 'NTMwNjA5NjgwODM0MTAxMjY4.D2BAnw.T6r2Sgrzoq-OjkGHkxZ4XSN4QDY'

client = discord.Client()

@client.event
async def on_message(message):
    d = {0: "Tomato Bacterial Spot",
        1: "Tomato Early blight",
        2: "Tomato healthy",
        4: "Tomato Leaf Mold",
        5: "Tomato Septoria leaf spot", 
        3: "Tomato Late blight", 
        6: "Tomato Two-spotted spider mite",  
        7: "Tomato Target Spot",
        8: "Tomato mosaic virus", 
        9: "Tomato Yellow Leaf Curl Virus"}
    if message.author == client.user:
        return
    #print(message.attachments)
    msg="nooo"
    async with aiohttp.ClientSession() as session:
        img_url = message.attachments[0]['proxy_url']
        async with session.get(img_url) as resp:
            f = await aiofiles.open('temp.png', mode='wb')
            await f.write(await resp.read())
            await f.close()
    filename = 'finalized_model.sav'
    infile = open(filename, 'rb')
    model = pickle.load(infile)
    msg = model.predict(np.asarray(create_dataset('file.png')).reshape(1, -1))
    msg=d[int(msg[0])]
    infile.close()
    await client.send_message(message.channel, msg)



@client.event
async def on_ready():
    print('Logged in as')
    print(client.user.name)
    print(client.user.id)
    print('------')

client.run(TOKEN)
