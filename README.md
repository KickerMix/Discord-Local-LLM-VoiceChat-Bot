# Discord Local LLM Voice Chat Bot

> **Disclaimer**  
> This project was entirely developed with the assistance of ChatGPT.  
> It is currently in a very early (alpha) stage of development.  
> Bugs, crashes, and unexpected behavior are likely to occur.  
> Use with caution and report any issues to help improve the project.  
>
> Tested with RTX 3060 (12GB) and RTX 4060 (8GB).  
> Running all of this on CUDA takes about >7GB VRAM and >6GB RAM (Depends on your LLM Model)

A Discord voice chat bot that uses OpenAI Whisper for real-time speech-to-text (STT), Coqui TTS for text-to-speech (TTS), and a local LLM (LM Studio) for conversational responses — all running locally.

## Features
- **Automatic voice channel join**: On startup, auto-connects to a voice channel with activity.
- **Real-time transcription**: Uses Whisper to transcribe user speech in the channel.
- **Keyword triggering**: Responds when predefined keywords are detected.
- **Conversational AI**: Queries a local LLM (via LM Studio API) for generating responses.
- **TTS modes**:
  - **default**: Uses a standard TTS voice for responses.
  - **clone**: Uses voice cloning from a sample `.wav` file.
- **Slash commands**:
  - `/join` — connect to your voice channel.
  - `/leave` — disconnect from the voice channel.
  - `/saya_tts [mode]` — switch between `default` and `clone` TTS modes.
  - `/saya_list` — browse and select a sample voice file for cloning.

## Requirements
- **Python 3.10+**
- **FFmpeg** installed on your system (for audio encoding/decoding).
- **CUDA 12.1** (for GPU acceleration)
- **CUDA Toolkit** (https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local)
- **cuDNN** (https://developer.nvidia.com/cudnn)
## Additional Requirements
- **cudnn_ops64_9.dll** (https://github.com/Purfview/whisper-standalone-win/releases/download/libs/cuBLAS.and.cuDNN_CUDA12_win_v2.7z)
- **^** Unpack cudnn_ops64_9.dll to "\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"

### Python dependencies
Listed in `requirements.txt`:
```
Also you need to install torch and torchaudio. For now it was tested only with:
torch 2.2.0+cu121
torchaudio 2.2.0+cu121
```
Example:

```
pip install torch==2.2.0+cu121 torchvision==0.17.0+cu121 torchaudio==2.2.0+cu121 --index-url https://download.pytorch.org/whl/cu121
```

## Installation
1. **Clone the repository**
   ```
   git clone https://github.com/KickerMix/Discord-Local-LLM-VoiceChat-Bot.git
   cd Discord-Local-LLM-VoiceChat-Bot
   ```
2. **Create and activate a virtual environment**
   ```
   python -m venv venv
   source venv/bin/activate   # Linux/macOS
   venv\Scripts\activate    # Windows
   ```
3. **Install dependencies**
   ```
   pip install -r requirements.txt
   
   # Also don't forget about torch cuda
   ```
4. **Install FFmpeg**
   ```
   # Ubuntu/Debian
   sudo apt update && sudo apt install ffmpeg

   # Windows: download from https://ffmpeg.org and
   # add to PATH
   ```

## Configuration
1. **Environment variables**: Create `.env` and set:
   ```ini
   DISCORD_TOKEN=YOUR_DISCORD_BOT_TOKEN
   LM_STUDIO_API_URL=http://127.0.0.1:5000    # or your LM Studio endpoint
   LANGUAGE=en
   ROLE=System prompt or role content for LLM
   KEYWORDS=hello,bot           # comma-separated keywords to trigger
   GUILD_ID=123456789012345678          # your Discord server ID
   TTS_MODE='default'
   SELECTED_FILE_PATH='./sample/sample_default.wav'
   ```

## Usage
1. **Start the bot**:
   ```
   # You can use bot.bat
   or
   python main.py
   ```
2. **Interact on Discord**:
   - Join a voice channel and use `/join` to have Saya auto-connect (if it's not connected on startup).
   - Speak in the channel; when a keyword is detected, Saya will record and process your speech.
   - Use `/saya_tts clone` and `/saya_list` to choose a custom voice sample for cloning.
   - Leave with `/leave`.

## Directory Structure
```
├── audio/                  # Recorded audio & generated responses at runtime
├── sample/                 # Sample .wav files for voice cloning
│   └── sample_default.wav  # Default cloning voice
├── .env                    # Settings for bot
├── main.py                 # Bot implementation
├── requirements.txt        # Python dependencies
├── bot_debug.log           # Debug logging
├── join.wav                # wav file for play on connecting to vc
├── trigger.wav             # wav file for play on keyword triggering start
├── listening.wav           # wav file for play on keyword triggering stop
├── .gitignore              # Ignored files & folders
```

## Known Issues
- The bot may occasionally disconnect from the voice channel. After disconnecting, it will stop functioning properly until fully restarted.
- Slash commands may sometimes respond slowly or fail to trigger.
- The bot may not always accurately detect keywords, especially in the presence of background noise or unclear speech.
- Under high system load, transcription and response generation may experience noticeable delays.
- Voice cloning mode may produce artifacts or degraded audio quality.
- Having multiple active speakers in the voice channel may cause confusion in speech processing and keyword detection.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
