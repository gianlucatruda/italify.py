# italify.py

Convert English text files into Italian Anki flashcard decks with AI translations and text-to-speech audio.

(I oneshot-prompted GPT-5 via the API with the highest reasoning level. I lightly edited and formatted the resulting script. It worked first time. Claude wrote the rest of this README. What a time to be alive.)

```bash
llm -m gpt-5 --no-stream -o reasoning_effort high "Oneshot-implement the
following simple script. It's called italify. It takes a plain text file as
input. The input has English text, with one concept per line (e.g. 'up or down:
I walk up the stairs. She runs down the stairs.'). The output is an Anki deck.
Each card in the anki deck has a front side with one line from the English text
input (e.g. 'up or down: I walk up the stairs. She runs down the stairs.') and
a back side with (1) the Italian text translation (in modern, conversational
Italian) and (2) an embedded audio clip of an Italian TTS voice speaking the
Italian translation. Use Elevenlabs' free tier to generate the audio, if it's
appropriate. Otherwise suggest viable alternatives. Use GPT-5 via the OpenAI
API (I have an API key) for the translations. Use Python to write the script,
using as few dependencies as possible (use uv frontmatter so they will
automatically be called when I run the script using script runner mode). I will
run 'italify.py english.txt Italian.apkg' and get a progress indicator with
lots of helpful feedback. Finally, Italian.apkg will be generated for me to
import into Anki 24+ with audio working."
```

---

## Overview

Italify is a Python script that takes a plain text file (one concept per line) and generates an Anki deck (.apkg) with:
- **English text** on the front of each card
- **Italian translation** on the back (powered by OpenAI GPT models)
- **Audio pronunciation** of the Italian text (via ElevenLabs or OpenAI TTS)

Perfect for creating Italian vocabulary decks, phrase collections, or study materials from any English text.

## Setup

### Prerequisites
- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### API Keys
You'll need at least one of these:
- **OpenAI API key** (required for translations, optional for TTS)
- **ElevenLabs API key** (recommended for high-quality Italian TTS)

### Environment Variables
```bash
# Required for translations
export OPENAI_API_KEY="your-openai-api-key"

# Optional: TTS with ElevenLabs (recommended)
export ELEVENLABS_API_KEY="your-elevenlabs-api-key"
export ELEVENLABS_VOICE_ID="your-voice-id"  # or use ELEVENLABS_VOICE_NAME

# Optional: Customize models
export OPENAI_MODEL="gpt-4o"  # or "gpt-5" if available
export TTS_PROVIDER="elevenlabs"  # or "openai" or "none"
```

## Usage

### Basic Usage
```bash
# With uv (handles dependencies automatically)
uv run italify.py input.txt output.apkg

# With pip (install dependencies first)
pip install openai genanki requests tqdm
python italify.py input.txt output.apkg
```

### Input Format
Create a text file with one concept per line:
```
Hello, how are you?
I would like a coffee, please.
Where is the bathroom?
up or down: I walk up the stairs. She runs down the stairs.
```

### Command Line Options
```bash
# Skip audio generation
uv run italify.py input.txt output.apkg --no-audio

# Force TTS provider
uv run italify.py input.txt output.apkg --tts-provider elevenlabs

# Use different OpenAI model
uv run italify.py input.txt output.apkg --model gpt-4o-mini

# Dry run (no API calls, for testing)
uv run italify.py input.txt output.apkg --dry-run
```

### TTS Providers
- **ElevenLabs** (recommended): High-quality Italian voices, free tier available
- **OpenAI TTS**: Good quality, uses your OpenAI credits
- **None**: Text-only cards without audio

### Import to Anki
1. Run the script to generate your `.apkg` file
2. Open Anki
3. File → Import → Select your `.apkg` file
4. Enable "Play audio automatically" in Anki preferences for best experience

## Examples

```bash
# Complete setup with ElevenLabs TTS
export OPENAI_API_KEY="sk-..."
export ELEVENLABS_API_KEY="..."
export ELEVENLABS_VOICE_NAME="Bella"
uv run italify.py phrases.txt Italian_Phrases.apkg

# OpenAI-only setup (translation + TTS)
export OPENAI_API_KEY="sk-..."
uv run italify.py --tts-provider openai vocab.txt Italian_Vocab.apkg

# Text-only cards (no audio)
export OPENAI_API_KEY="sk-..."
uv run italify.py --no-audio sentences.txt Italian_Sentences.apkg
```

The resulting Anki deck will have cards with English on the front, Italian translation and audio on the back.

