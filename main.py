import os
import json
import asyncio
import logging
import re
from pathlib import Path
import interactions
from interactions import (
    Client,
    slash_command,
    SlashContext,
    SlashCommandOption,
    SlashCommandChoice,
    OptionType,
    Intents,
    Button,
    ButtonStyle,
    ComponentContext,
    Embed,
)
from interactions.api.events import Startup
from interactions.api.voice.audio import AudioVolume
from faster_whisper import WhisperModel
from TTS.api import TTS
from dotenv import load_dotenv
import aiohttp
from datetime import datetime

# ── Logging configuration ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bot_debug.log", encoding="utf-8")
    ]
)
logger = logging.getLogger(__name__)

# ── Settings file path and defaults ───────────────────────────────────────────
SETTINGS_PATH = Path("settings.json")
DEFAULT_SETTINGS = {
    "TTS_MODE": "default",
    "selected_file_path": "./sample/sample_miside.wav"
}

def load_settings():
    if SETTINGS_PATH.exists():
        try:
            return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("settings.json is corrupted, overwriting with default settings")
    save_settings(DEFAULT_SETTINGS)
    return DEFAULT_SETTINGS.copy()

def save_settings(settings: dict):
    SETTINGS_PATH.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")

# ── Loading and initializing settings ─────────────────────────────────────────
settings = load_settings()
TTS_MODE = settings["TTS_MODE"]
current_speaker_wav = settings["selected_file_path"]

# ── Environment variables and constants ────────────────────────────────────────
load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")
LM_STUDIO_API_URL = os.getenv("LM_STUDIO_API_URL", "http://127.0.0.1:5000")
ROLE = os.getenv("ROLE", "")
KEYWORDS = [w.strip() for w in os.getenv("KEYWORDS", "").split(",") if w.strip()]

AUDIO_DIR = Path(os.path.abspath("./audio"))
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
GUILD_ID = 188965959707525120  # replace with your guild id

# Compile regex for keywords
pattern = (r"\b(?:" + "|".join(re.escape(w) for w in KEYWORDS) + r")\b") if KEYWORDS else None
KEYWORD_RE = re.compile(pattern, flags=re.IGNORECASE) if pattern else None

# ── Initializing Whisper and TTS ────────────────────────────────────────────────
stt_model = WhisperModel(
    model_size_or_path="large-v2",
    device="cuda",
    compute_type="int8"
)
tts = TTS(model_name="multilingual/multi-dataset/xtts_v2")
tts.to("cuda")

# ── Main bot with voice recording ────────────────────────────────────────────────
intents = Intents.ALL
bot = interactions.Client(token=TOKEN, intents=intents)

recording_lock = asyncio.Lock()
is_connected = False
is_playing_response = False
current_channel = None

@bot.listen(Startup)
async def on_startup():
    global is_connected  # explicitly modify the global variable
    is_connected = False  # reset connection state on startup

    guild = await bot.fetch_guild(GUILD_ID)
    all_states = guild.voice_states

    # Find a channel with exactly one member
    for channel in guild.channels:
        if channel.type is not interactions.ChannelType.GUILD_VOICE:
            continue
        members_states = [vs for vs in all_states if vs.channel and vs.channel.id == channel.id]
        if len(members_states) == 1:  # Channel with one person
            vs_state = members_states[0]
            member = vs_state.member
            nick = member.display_name
            voice = await channel.connect()
            logger.info(f"Auto-connecting to {nick} in channel {channel.name}")
            prompt = f'Greet the user "{nick}" and introduce yourself as "Saya".'
            async with aiohttp.ClientSession() as sess:
                payload = {
                    "model": "your-model-id",
                    "messages": [{"role": "system", "content": ROLE}, {"role": "user", "content": prompt}],
                    "max_tokens": 350
                }
                resp = await sess.post(f"{LM_STUDIO_API_URL}/v1/chat/completions", json=payload)
                data = await resp.json()
            greeting = data["choices"][0]["message"]["content"]
            ts = datetime.now().strftime("%Y%m%d%H%M%S")
            out_path = AUDIO_DIR / f"greet_{ts}.wav"
            await asyncio.to_thread(
                tts.tts_to_file,
                text=greeting,
                speaker="Ana Florence",
                language="en",
                file_path=str(out_path)
            )
            await voice.play(AudioVolume(str(out_path)))
            await asyncio.sleep(2)
            await voice.play(AudioVolume("join.wav"))
            asyncio.create_task(start_recording(voice, str(AUDIO_DIR)))
            break
    else:
        # If no single-person channel is found, search for a channel with multiple people
        for channel in guild.channels:
            if channel.type is not interactions.ChannelType.GUILD_VOICE:
                continue
            members_states = [vs for vs in all_states if vs.channel and vs.channel.id == channel.id]
            if len(members_states) > 1:  # Channel with multiple people
                voice = await channel.connect()
                logger.info(f"Auto-connecting to channel {channel.name} with multiple people.")
                await asyncio.sleep(2)  # Wait briefly before starting to listen
                await voice.play(AudioVolume("join.wav"))
                asyncio.create_task(start_recording(voice, str(AUDIO_DIR)))
                break

    logger.info("Bot is now waiting for keywords.")

async def reconnect_voice(channel):
    try:
        if channel:
            return await channel.connect()
        logger.error("Channel not found for reconnect.")
    except Exception as e:
        logger.error(f"Error reconnecting to voice channel: {e}")
    return None

async def start_recording(voice_state, output_dir: str):
    global current_channel
    if not voice_state:
        voice_state = await reconnect_voice(current_channel) if current_channel else None
        if not voice_state:
            logger.error("Failed to restore connection.")
            return
    async with recording_lock:
        await voice_state.start_recording(output_dir=output_dir, encoding="wav")
    await asyncio.sleep(3)
    await stop_recording(voice_state)

    for user_id, file in list(voice_state.recorder.output.items()):
        audio_path = file.get("path") if isinstance(file, dict) else file
        if os.path.exists(audio_path):
            await process_audio_for_user(voice_state, user_id, audio_path)

    if not is_playing_response:
        await start_recording(voice_state, output_dir)

async def stop_recording(voice_state):
    async with recording_lock:
        await voice_state.stop_recording()

async def transcribe_audio(audio_path: str) -> str:
    def _transcribe_blocking(path):
        segments, info = stt_model.transcribe(path, language="ru", beam_size=5)
        return segments, info

    try:
        segments, info = await asyncio.to_thread(_transcribe_blocking, audio_path)
        texts = [seg.text for seg in segments if hasattr(seg, 'text')]
        logger.debug(f"Detected language '{info.language}' (p={info.language_probability:.2f})")
        return " ".join(texts)
    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        return ""

async def process_audio_for_user(voice_state, user_id: int, audio_path: str):
    global is_playing_response
    full_text = await transcribe_audio(audio_path)
    logger.debug(f"Text: {full_text}")
    if KEYWORD_RE and KEYWORD_RE.search(full_text):
        is_playing_response = True
        await voice_state.play(AudioVolume("trigger.wav"))
        await process_audio_for_individual_user(
            voice_state,
            user_id,
            full_text,
            speaker_wav=current_speaker_wav
        )

async def process_audio_for_individual_user(voice_state, user_id, initial_text, speaker_wav: str):
    global is_playing_response, current_channel
    # Clear old recordings
    for old in AUDIO_DIR.rglob("*.wav"):
        old.unlink()
    voice_state.recorder.output.clear()

    now = datetime.now().strftime("%Y%m%d%H%M%S")
    user_dir = AUDIO_DIR / f"individual_{user_id}_{now}"
    user_dir.mkdir(parents=True, exist_ok=True)

    await voice_state.start_recording(output_dir=str(user_dir), encoding="wav")
    await asyncio.sleep(10)
    await stop_recording(voice_state)

    if not voice_state:
        voice_state = await reconnect_voice(current_channel) if current_channel else None
        if not voice_state:
            logger.error("Failed to restore connection.")
            return

    await voice_state.play(AudioVolume("listening.wav"))
    data = voice_state.recorder.output.get(user_id)
    audio_path = data.get("path") if isinstance(data, dict) else data
    if not audio_path or not os.path.exists(audio_path):
        logger.error(f"Audio file does not exist: {audio_path}")
        return

    followup = await transcribe_audio(audio_path)
    full_text = f"{initial_text} {followup}".strip()
    logger.debug(f"Text: {followup}")

    if TTS_MODE == "default":
        await generate_and_play_response(voice_state, user_id, full_text)
    else:
        await generate_and_play_voiceclone_response(
            voice_state, user_id, full_text, speaker_wav
        )

async def generate_and_play_response(voice_state, user_id, text: str):
    """
    Pipeline TTS responses one sentence at a time to avoid GPU overload.
    """
    global is_playing_response
    # 1) Get LLM answer
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    async with aiohttp.ClientSession() as session:
        payload = {
            "model": "your-model-id",
            "messages": [{"role": "system", "content": ROLE}, {"role": "user", "content": text}],
            "max_tokens": 1000
        }
        resp = await session.post(f"{LM_STUDIO_API_URL}/v1/chat/completions", json=payload)
        data = await resp.json()
        answer = data["choices"][0]["message"]["content"]

    # 2) Split into sentences
    sentences = re.split(r'(?<=[\.!\?])\s+', answer.strip())
    if not sentences:
        return

    # Helper: run TTS for one sentence
    def tts_to_file(sentence, idx):
        out = AUDIO_DIR / f"response_{ts}_{idx}.wav"
        tts.tts_to_file(
            text=sentence,
            speaker="Ana Florence",
            language="en",
            file_path=str(out)
        )
        return str(out)

    is_playing_response = True

    # 3) Generate and play first sentence synchronously
    first_wav = await asyncio.to_thread(tts_to_file, sentences[0], 0)
    play_task = asyncio.create_task(voice_state.play(AudioVolume(first_wav)))

    # 4) Now pipeline each remaining sentence one by one
    for idx, sent in enumerate(sentences[1:], start=1):
        # While the previous is still playing, generate this one on the side
        next_wav = await asyncio.to_thread(tts_to_file, sent, idx)

        # Wait for the previous playback to finish
        await play_task

        # Play this sentence, then loop to generate+queue the next
        play_task = asyncio.create_task(voice_state.play(AudioVolume(next_wav)))

    # 5) Wait for the final sentence to finish
    await play_task
    is_playing_response = False

async def generate_and_play_voiceclone_response(voice_state, user_id, text: str, speaker_wav: str):
    """
    Pipeline TTS-clone responses one sentence at a time to avoid GPU overload.
    """
    global is_playing_response
    # 1) Get LLM answer
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    async with aiohttp.ClientSession() as session:
        payload = {
            "model": "your-model-id",
            "messages": [{"role": "system", "content": ROLE}, {"role": "user", "content": text}],
            "max_tokens": 1000
        }
        resp = await session.post(f"{LM_STUDIO_API_URL}/v1/chat/completions", json=payload)
        data = await resp.json()
        answer = data["choices"][0]["message"]["content"]

    # 2) Split into sentences
    sentences = re.split(r'(?<=[\.!\?])\s+', answer.strip())
    if not sentences:
        return

    # Helper: run TTS-clone for one sentence
    def tts_clone_to_file(sentence, idx):
        out = AUDIO_DIR / f"clone_{ts}_{idx}.wav"
        tts.tts_to_file(
            text=sentence,
            speaker_wav=speaker_wav,
            language="en",
            file_path=str(out)
        )
        return str(out)

    is_playing_response = True

    # 3) Generate and play first sentence synchronously
    first_wav = await asyncio.to_thread(tts_clone_to_file, sentences[0], 0)
    play_task = asyncio.create_task(voice_state.play(AudioVolume(first_wav)))

    # 4) Now pipeline each remaining sentence one by one
    for idx, sent in enumerate(sentences[1:], start=1):
        # While the previous is still playing, generate this one on the side
        next_wav = await asyncio.to_thread(tts_clone_to_file, sent, idx)

        # Wait for the previous playback to finish
        await play_task

        # Play this sentence, then loop to generate+queue the next
        play_task = asyncio.create_task(voice_state.play(AudioVolume(next_wav)))

    # 5) Wait for the final sentence to finish
    await play_task
    is_playing_response = False

# ── Slash commands ─────────────────────────────────────────────────────────────
@slash_command(
    name="join",
    description="Connect to a voice channel."
)
async def join(ctx: SlashContext):
    global is_connected, current_channel
    if is_connected:
        return await ctx.send("I'm already connected.")
    if not ctx.author.voice:
        return await ctx.send("❗ Please join a voice channel and try again.")
    vs = await ctx.author.voice.channel.connect()
    is_connected = True
    current_channel = ctx.author.voice.channel
    await ctx.send("✅ Connected.")
    asyncio.create_task(start_recording(vs, str(AUDIO_DIR)))

@slash_command(
    name="leave",
    description="Disconnect from the voice channel."
)
async def leave(ctx: SlashContext):
    global is_connected, current_channel
    if ctx.voice_state:
        await ctx.voice_state.disconnect()
        is_connected = False
        current_channel = None
        await ctx.send("🛑 Disconnected.")
    else:
        await ctx.send("🤷 I'm not connected.")

@slash_command(
    name="saya_tts",
    description="Switch TTS mode: default or clone",
    options=[
        SlashCommandOption(
            name="mode",
            description="Select TTS mode",
            required=True,
            type=OptionType.STRING,
            choices=[
                SlashCommandChoice(name="default", value="default"),
                SlashCommandChoice(name="clone",   value="clone"),
            ],
        )
    ]
)
async def saya_tts(ctx: SlashContext, mode: str):
    global TTS_MODE, settings
    if mode not in ("default", "clone"):
        return await ctx.send("❗ Invalid mode, choose 'default' or 'clone'.", ephemeral=True)
    TTS_MODE = mode
    settings["TTS_MODE"] = TTS_MODE
    save_settings(settings)
    await ctx.send(f"✅ TTS mode set to: `{mode}`")

# ── Pagination for file list ─────────────────────────────────────────────────
def generate_file_list_page(files, page=1, items_per_page=5):
    start_idx = (page - 1) * items_per_page
    end_idx = page * items_per_page
    selected_files = files[start_idx:end_idx]

    embed = Embed(
        title="List of available sample files",
        description="Click the button with the file number to select."
    )
    for idx, fname in enumerate(selected_files, start=start_idx + 1):
        embed.add_field(name=f"{idx}. {fname}", value=fname, inline=False)

    components = []
    if page > 1:
        components.append(Button(label="« Back", custom_id="prev_page", style=ButtonStyle.SECONDARY))
    if end_idx < len(files):
        components.append(Button(label="Next »", custom_id="next_page", style=ButtonStyle.SECONDARY))

    for idx, fname in enumerate(selected_files, start=start_idx + 1):
        components.append(Button(
            label=str(idx),
            custom_id=f"select_{fname}",
            style=ButtonStyle.PRIMARY
        ))

    return embed, components

@slash_command(name="saya_list", description="Show the list of files and select one.")
async def saya_list(ctx: SlashContext):
    global current_speaker_wav, settings

    sample_files = [f.name for f in Path("./sample").glob("*.wav")]
    if not sample_files:
        return await ctx.send("❗ No `.wav` files found in the `./sample` folder.")

    page = 1
    embed, components = generate_file_list_page(sample_files, page)
    await ctx.send(embed=embed, components=components, ephemeral=True)

    while True:
        used = await bot.wait_for_component(
            components=components,
            check=lambda c: c.ctx.author.id == ctx.author.id,
            timeout=60
        )

        cid = used.ctx.custom_id
        if cid == "prev_page":
            page -= 1
        elif cid == "next_page":
            page += 1
        elif cid.startswith("select_"):
            fname = cid.split("_", 1)[1]
            current_speaker_wav = f"./sample/{fname}"
            settings["selected_file_path"] = current_speaker_wav
            save_settings(settings)

            # Edit original and remove buttons
            await used.ctx.edit_origin(
                embed=Embed(
                    title="Done! 🎉",
                    description=f"Selected file for cloning: `{fname}`"
                ),
                components=[]
            )
            # Send separate ephemeral confirmation if desired
            await used.ctx.send("Configuration saved ✅", ephemeral=True)
            return

        embed, components = generate_file_list_page(sample_files, page)
        await used.ctx.edit_origin(embed=embed, components=components)

bot.start()