import os
import discord
import asyncio
import re
import datetime
from discord import ui, app_commands
from discord.ext import commands
from dotenv import load_dotenv
from api.main import DiaryProcessor, NLPUtils, OpenAIUtils  # Import your utilities from API

# Load environment variables
load_dotenv()

DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Set up intents and bot client
intents = discord.Intents.default()
intents.message_content = True
client = commands.Bot(command_prefix='!', intents=intents, case_insensitive=True)

# Initialize API utilities
openai_util = OpenAIUtils(OPENAI_API_KEY)
nlp_util = NLPUtils()
processor = DiaryProcessor(openai_util, nlp_util)

start_time = datetime.datetime.now(datetime.timezone.utc)

# Embed for diary processing in progress
processing_embed = discord.Embed(
    title="Processing Diary Entry",
    description="Processing your diary entry. This may take a moment...",
    color=discord.Color.orange()
)

# Mock function to simulate image generation and song recommendation
async def process_diary_entry(entry: str):
    await asyncio.sleep(2)  # Simulate processing time

    # Mock Data
    image_url = "https://example.com/mock_image.png"
    song_url = "https://open.spotify.com/track/mock_song_id"
    
    return image_url, song_url


# Button to retry diary processing
class RetryButton(ui.View):
    def __init__(self, entry, user_id) -> None:
        super().__init__()
        self.entry = entry
        self.user_id = user_id

    @discord.ui.button(label='Retry', emoji="♻️", style=discord.ButtonStyle.green)
    async def retry(self, interaction: discord.Interaction, button: ui.Button):
        if interaction.user.id != self.user_id:
            await interaction.response.send_message("You don't own this command.", ephemeral=True)
            return

        button.disabled = True
        await interaction.response.edit_message(embed=processing_embed, view=self)
        image_url, song_url = await process_diary_entry(self.entry)

        if image_url and song_url:
            embed = discord.Embed(title="Diary Processed", color=discord.Color.blue())
            embed.set_image(url=image_url)
            embed.add_field(name="Listen to this song", value=f"[Spotify Link]({song_url})", inline=False)
            
            button.disabled = False
            await interaction.edit_original_response(embed=embed, view=self)
        else:
            await interaction.response.edit_message(content="Failed to process diary entry.", embed=None)


# Command to process a diary entry and generate image/song recommendation
@client.command(name="diary")
async def diary(ctx, *, entry: str):
    message = await ctx.send(embed=processing_embed)

    # Simulate the diary processing and generate image and song (mock data for now)
    image_url, song_url = await process_diary_entry(entry)

    if image_url and song_url:
        view = RetryButton(entry, user_id=ctx.author.id)
        embed = discord.Embed(title="Diary Processed", color=discord.Color.blue())
        embed.set_image(url=image_url)
        embed.add_field(name="Listen to this song", value=f"[Spotify Link]({song_url})", inline=False)

        await message.edit(embed=embed, view=view)
    else:
        await message.edit(content="Failed to process diary entry.", embed=None)


# Custom info command to list all available bot commands
@client.command(name="info", help="Lists all available bot commands")
async def info(ctx):
    embed = discord.Embed(title="Info - Available Commands", color=discord.Color.green())

    # List of commands with descriptions
    embed.add_field(
        name="!diary <your_entry>",
        value="Process your diary entry, generate an image, and recommend a song.",
        inline=False
    )
    embed.add_field(
        name="!stats",
        value="Display bot statistics, such as uptime, server count, and member count.",
        inline=False
    )
    embed.add_field(
        name="!info",
        value="Show this info message with all available commands.",
        inline=False
    )

    await ctx.send(embed=embed)


# Bot statistics command (fixing 'tree' issue by using @client.command)
@client.command(name="stats", help="Get bot statistics")
async def stats(ctx):
    total_servers = len(client.guilds)
    total_members = sum(guild.member_count for guild in client.guilds)
    uptime = datetime.datetime.now(datetime.timezone.utc) - start_time

    embed = discord.Embed(
        title="Bot Statistics",
        description=f"Servers: {total_servers}\nMembers: {total_members}\nUptime: {uptime}",
        color=discord.Color.blue()
    )
    await ctx.send(embed=embed)


# On bot ready
@client.event
async def on_ready():
    print(f'Logged in as {client.user.name}')
    await client.tree.sync()


# On message mention
@client.event
async def on_message(message):
    if message.author == client.user:
        return

    # Respond to mention with info about the !info command
    if client.user.mention in message.content:
        embed = discord.Embed(
            title="Bot is Ready!",
            description="You can use the `!info` command to see a list of available commands.",
            color=discord.Color.blue()
        )
        await message.channel.send(embed=embed)
    
    await client.process_commands(message)


client.run(DISCORD_TOKEN)
