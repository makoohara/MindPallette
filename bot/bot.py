import os
import discord
import asyncio
import re
import datetime
import aiohttpy
from discord import ui, app_commands
from discord.ext import commands
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
API_URL = os.getenv('API_URL')  # Add this to your .env file

# Set up intents and bot client
intents = discord.Intents.default()
intents.message_content = True
client = commands.Bot(command_prefix='!', intents=intents, case_insensitive=True)

start_time = datetime.datetime.now(datetime.timezone.utc)

# Embed for diary processing in progress
processing_embed = discord.Embed(
    title="Processing Diary Entry",
    description="Processing your diary entry. This may take a moment...",
    color=discord.Color.orange()
)

# Function to process diary entry through API
async def process_diary_entry(entry: str):
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(f"{API_URL}/api/home", json={"mood": entry}) as response:
                if response.status == 200:
                    data = await response.json()
                    # Extract image URL and song from API response
                    image_urls = data.get('img_urls', [])
                    song = data.get('song', '')
                    
                    # Return the first image URL (or None) and the song
                    return image_urls[0] if image_urls else None, song
                else:
                    print(f"API Error: Status {response.status}")
                    return None, None
        except Exception as e:
            print(f"Error processing diary entry: {e}")
            return None, None

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
        image_url, song = await process_diary_entry(self.entry)

        if image_url and song:
            embed = discord.Embed(title="Diary Processed", color=discord.Color.blue())
            embed.set_image(url=image_url)
            embed.add_field(name="Recommended Song", value=song, inline=False)
            
            button.disabled = False
            await interaction.edit_original_response(embed=embed, view=self)
        else:
            await interaction.edit_original_response(content="Failed to process diary entry.", embed=None)

# Command to process a diary entry
@client.command(name="diary")
async def diary(ctx, *, entry: str):
    message = await ctx.send(embed=processing_embed)

    # Process the diary entry through the API
    image_url, song = await process_diary_entry(entry)

    if image_url and song:
        view = RetryButton(entry, user_id=ctx.author.id)
        embed = discord.Embed(title="Diary Processed", color=discord.Color.blue())
        embed.set_image(url=image_url)
        embed.add_field(name="Recommended Song", value=song, inline=False)

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
