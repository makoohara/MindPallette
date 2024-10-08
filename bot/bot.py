import os
import discord
from dotenv import load_dotenv
from your_api_module import DiaryProcessor, NLPUtils, OpenAIUtils  # Import necessary classes from the API code

# Load environment variables from .env file
load_dotenv()

TOKEN = os.getenv('DISCORD_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize discord client
intents = discord.Intents.default()
intents.messages = True
client = discord.Client(intents=intents)

# Initialize API utilities (using the classes already defined in your API code)
openai_util = OpenAIUtils(OPENAI_API_KEY)
nlp_util = NLPUtils()
processor = DiaryProcessor(openai_util, nlp_util)

# When the bot is ready
@client.event
async def on_ready():
    print(f'Bot has logged in as {client.user}')

# On receiving a message
@client.event
async def on_message(message):
    if message.author == client.user:
        return  # Prevents bot from replying to its own messages

    # Example command for mood analysis
    if message.content.startswith('!mood'):
        entry = message.content[len('!mood '):]
        await message.channel.send('Processing your mood entry...')

        # For now, using mock data until OpenAI API is connected
        # Mock data
        processed_data = processor.data_process(entry)  # Replace with actual or mock data process
        mock_image_attributes = processed_data[1]['image_attributes']  # Example mock
        mock_song = "Mock Song by Mock Artist"

        # Send response
        await message.channel.send(f'Mock Image Attributes: {mock_image_attributes}')
        await message.channel.send(f'Mock Song Recommendation: {mock_song}')

    # Example command for song recommendation
    elif message.content.startswith('!song'):
        entry = message.content[len('!song '):]
        await message.channel.send('Fetching a song recommendation...')
        song = openai_util.recommend_song(entry)  # Use real or mock data for now
        await message.channel.send(f'Recommended Song: {song}')

# Run the bot
client.run(TOKEN)
