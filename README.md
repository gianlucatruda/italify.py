# italify.py

```bash
llm -m gpt-5 --no-stream -o reasoning_effort high "Oneshot-implement the following simple script. It's called italify. It takes a plain text file as input. The input has English text, with one concept per line (e.g. 'up or down: I walk up the stairs. She runs down the stairs.'). The output is an Anki deck. Each card in the anki deck has a front side with one line from the English text input (e.g. 'up or down: I walk up the stairs. She runs down the stairs.') and a back side with (1) the Italian text translation (in modern, conversational Italian) and (2) an embedded audio clip of an Italian TTS voice speaking the Italian translation. Use Elevenlabs' free tier to generate the audio, if it's appropriate. Otherwise suggest viable alternatives. Use GPT-5 via the OpenAI API (I have an API key) for the translations. Use Python to write the script, using as few dependencies as possible (use uv frontmatter so they will automatically be called when I run the script using script runner mode). I will run 'italify.py english.txt Italian.apkg' and get a progress indicator with lots of helpful feedback. Finally, Italian.apkg will be generated for me to import into Anki 24+ with audio working." > $(ecdt).md
```

Here’s a single-file script you can drop in as italify.py. It uses OpenAI for translation, ElevenLabs for TTS when available, and falls back to OpenAI TTS if ElevenLabs isn’t configured. It builds a .apkg you can import into Anki 24+ with working audio. It uses uv script frontmatter so dependencies auto-install when you run with uv.

Save as italify.py, make it executable, then run:
- uv run italify.py english.txt Italian.apkg
Or:
- python italify.py english.txt Italian.apkg
(uv is recommended for auto dependency handling.)

Before running, set at least:
- export OPENAI_API_KEY=your_openai_key
Optional for ElevenLabs audio:
- export ELEVENLABS_API_KEY=your_elevenlabs_key
- export ELEVENLABS_VOICE_ID=voice_id_here  # recommended
Or set TTS_PROVIDER=openai to use OpenAI TTS.

Notes:
- Translations: Defaults to OpenAI model gpt-4o. If you have access to a newer model (e.g., GPT-5 when available), set environment variable OPENAI_MODEL=gpt-5 or use --model gpt-5.
- TTS:
  - Preferred: ElevenLabs free tier. Set ELEVENLABS_API_KEY and preferably ELEVENLABS_VOICE_ID. You can also set ELEVENLABS_VOICE_NAME to auto-resolve an ID (e.g., “Vittoria”, “Bella”).
  - Fallback: OpenAI TTS (set OPENAI_API_KEY). Uses gpt-4o-mini-tts by default and voice "alloy".
  - Force provider with --tts-provider elevenlabs or --tts-provider openai.
- Usage examples:
  - uv run italify.py english.txt Italian.apkg
  - OPENAI_MODEL=gpt-4o TTS_PROVIDER=elevenlabs ELEVENLABS_API_KEY=... ELEVENLABS_VOICE_ID=... uv run italify.py english.txt Italian.apkg
  - TTS via OpenAI: TTS_PROVIDER=openai OPENAI_API_KEY=... uv run italify.py english.txt Italian.apkg
- Input format: one concept per line, e.g.:
  up or down: I walk up the stairs. She runs down the stairs.

The deck will have:
- Front: original English line
- Back: Italian translation + an audio button that plays the Italian TTS audio.
