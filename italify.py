#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "openai>=1.43.0",
#     "genanki>=0.13.1",
#     "requests>=2.32.3",
#     "tqdm>=4.66.4",
# ]
# ///
import argparse
import hashlib
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import requests
from tqdm import tqdm

try:
    from openai import OpenAI
except Exception as e:
    print(
        "Failed to import openai. If you are not using uv, install dependencies first: pip install openai genanki requests tqdm"
    )
    raise

try:
    import genanki
except Exception as e:
    print(
        "Failed to import genanki. If you are not using uv, install dependencies first: pip install genanki"
    )
    raise


# ---------------------------
# Helpers
# ---------------------------


def eprint(*a, **k):
    print(*a, file=sys.stderr, **k)


def sha1_hex(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def stable_64bit_int(name: str) -> int:
    # Make a deterministic 64-bit ID from a string
    h = hashlib.sha1(name.encode("utf-8")).hexdigest()[:16]
    return int(h, 16) & ((1 << 63) - 1)


def sanitize_filename(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_\-\.]+", "_", s.strip())
    s = s.strip("._")
    return s or "audio"


def read_lines(path: Path) -> List[str]:
    lines = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if not t:
                continue
            # Allow comments if user adds them
            if t.startswith("#"):
                continue
            lines.append(t)
    return lines


# ---------------------------
# Translation (OpenAI)
# ---------------------------


def translate_lines_openai(
    lines: List[str], client: OpenAI, model: str, dry_run: bool = False
) -> List[str]:
    translations: List[str] = []
    eprint(f"Translator model: {model}")
    for line in tqdm(lines, desc="Translating to Italian", unit="line"):
        if dry_run:
            translations.append(line)  # echo for testing
            continue

        for attempt in range(4):
            try:
                # Use low temperature for reliable translations
                resp = client.chat.completions.create(
                    model=model,
                    temperature=0.2,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a professional English→Italian translator. "
                                "Translate the user message into modern, conversational Italian that sounds natural. "
                                "Preserve meaning; keep names, lists, and formatting. "
                                "Do not add explanations. Output only the Italian translation, no quotes or extra text."
                            ),
                        },
                        {"role": "user", "content": line},
                    ],
                )
                translated = (resp.choices[0].message.content or "").strip()
                # Basic guard: ensure we didn't get empty/English back
                if not translated:
                    raise RuntimeError("Empty translation")
                translations.append(translated)
                break
            except Exception as ex:
                wait = 2**attempt
                if attempt < 3:
                    eprint(
                        f"Translation error (attempt {attempt + 1}/4): {ex}. Retrying in {wait}s..."
                    )
                    time.sleep(wait)
                else:
                    eprint(
                        f"Final translation error, using original text as fallback for this line: {ex}"
                    )
                    translations.append(line)
    return translations


# ---------------------------
# TTS Providers
# ---------------------------


class TTSProvider:
    def synthesize(self, text: str, outfile: Path) -> bool:
        raise NotImplementedError


class ElevenLabsTTS(TTSProvider):
    def __init__(
        self,
        api_key: str,
        voice_id: Optional[str],
        voice_name: Optional[str],
        model_id: str = "eleven_multilingual_v2",
    ):
        self.api_key = api_key
        self.voice_id = voice_id
        self.voice_name = voice_name
        self.model_id = model_id

        if not self.voice_id and self.voice_name:
            # Try to resolve voice id by name
            try:
                vid = self._resolve_voice_id_by_name(self.voice_name)
                if vid:
                    self.voice_id = vid
                    eprint(f"ElevenLabs voice '{self.voice_name}' resolved to id: {vid}")
                else:
                    eprint(f"Warning: Could not resolve ElevenLabs voice name '{self.voice_name}'.")
            except Exception as e:
                eprint(f"Warning: Failed to resolve ElevenLabs voice name '{self.voice_name}': {e}")

        if not self.voice_id:
            eprint(
                "Warning: ELEVENLABS_VOICE_ID not set. "
                "You can set ELEVENLABS_VOICE_ID or ELEVENLABS_VOICE_NAME. "
                "Attempting to pick a likely Italian-capable premade voice..."
            )
            try:
                self.voice_id = self._pick_default_voice_id()
                if self.voice_id:
                    eprint(f"Selected ElevenLabs voice id: {self.voice_id}")
            except Exception as e:
                eprint(f"Warning: Could not select a default ElevenLabs voice automatically: {e}")

    def _resolve_voice_id_by_name(self, name: str) -> Optional[str]:
        url = "https://api.elevenlabs.io/v1/voices"
        headers = {"xi-api-key": self.api_key}
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()
        for v in data.get("voices", []):
            if (v.get("name") or "").strip().lower() == name.strip().lower():
                return v.get("voice_id")
        return None

    def _pick_default_voice_id(self) -> Optional[str]:
        # Best-effort: pick a premade voice often available and multi-lingual.
        url = "https://api.elevenlabs.io/v1/voices"
        headers = {"xi-api-key": self.api_key}
        r = requests.get(url, headers=headers, timeout=30)
        if r.status_code != 200:
            return None
        data = r.json()
        candidates = []
        for v in data.get("voices", []):
            nm = (v.get("name") or "").lower()
            # Common premade voices that typically handle multiple languages well
            if nm in (
                "bella",
                "rachel",
                "antoni",
                "domi",
                "elli",
                "josh",
                "clyde",
                "mimi",
                "giovanni",
                "valentina",
                "serena",
                "matteo",
                "federico",
                "vittoria",
            ):
                candidates.append(v.get("voice_id"))
        return (
            candidates[0]
            if candidates
            else (data.get("voices", [{}])[0].get("voice_id") if data.get("voices") else None)
        )

    def synthesize(self, text: str, outfile: Path) -> bool:
        if not self.voice_id:
            eprint("ElevenLabs TTS: No voice_id available.")
            return False
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}"
        headers = {
            "xi-api-key": self.api_key,
            "accept": "audio/mpeg",
            "content-type": "application/json",
        }
        payload = {
            "text": text,
            "model_id": self.model_id,
            "voice_settings": {
                "stability": 0.35,
                "similarity_boost": 0.75,
                "style": 0.0,
                "use_speaker_boost": True,
            },
        }
        try:
            with requests.post(url, headers=headers, json=payload, stream=True, timeout=90) as r:
                if r.status_code != 200:
                    eprint(f"ElevenLabs TTS error {r.status_code}: {r.text[:200]}")
                    return False
                with outfile.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=64 * 1024):
                        if chunk:
                            f.write(chunk)
            return True
        except Exception as e:
            eprint(f"ElevenLabs TTS exception: {e}")
            return False


class OpenAITTS(TTSProvider):
    def __init__(self, api_key: str, voice: str = "alloy", model: str = "gpt-4o-mini-tts"):
        self.api_key = api_key
        self.voice = voice
        self.model = model

    def synthesize(self, text: str, outfile: Path) -> bool:
        url = "https://api.openai.com/v1/audio/speech"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "voice": self.voice,
            "input": text,
            "format": "mp3",
        }
        try:
            with requests.post(url, headers=headers, json=payload, stream=True, timeout=90) as r:
                if r.status_code != 200:
                    try:
                        err = r.json()
                    except Exception:
                        err = r.text
                    eprint(f"OpenAI TTS error {r.status_code}: {err}")
                    return False
                with outfile.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=64 * 1024):
                        if chunk:
                            f.write(chunk)
            return True
        except Exception as e:
            eprint(f"OpenAI TTS exception: {e}")
            return False


def choose_tts_provider(args, env: Dict[str, str]):
    forced = (args.tts_provider or env.get("TTS_PROVIDER") or "").strip().lower()
    el_key = env.get("ELEVENLABS_API_KEY")
    oa_key = env.get("OPENAI_API_KEY")

    if forced in ("elevenlabs", "11labs"):
        if not el_key:
            eprint("TTS provider forced to ElevenLabs but ELEVENLABS_API_KEY is missing.")
            return None, "none"
        return ElevenLabsTTS(
            el_key,
            args.elevenlabs_voice_id or env.get("ELEVENLABS_VOICE_ID"),
            args.elevenlabs_voice_name or env.get("ELEVENLABS_VOICE_NAME"),
            args.elevenlabs_model or env.get("ELEVENLABS_MODEL") or "eleven_multilingual_v2",
        ), "elevenlabs"

    if forced in ("openai", "oai"):
        if not oa_key:
            eprint("TTS provider forced to OpenAI but OPENAI_API_KEY is missing.")
            return None, "none"
        return OpenAITTS(
            oa_key,
            args.openai_voice or env.get("OPENAI_TTS_VOICE") or "alloy",
            args.openai_tts_model or env.get("OPENAI_TTS_MODEL") or "gpt-4o-mini-tts",
        ), "openai"

    # Auto-pick
    if el_key:
        return ElevenLabsTTS(
            el_key,
            args.elevenlabs_voice_id or env.get("ELEVENLABS_VOICE_ID"),
            args.elevenlabs_voice_name or env.get("ELEVENLABS_VOICE_NAME"),
            args.elevenlabs_model or env.get("ELEVENLABS_MODEL") or "eleven_multilingual_v2",
        ), "elevenlabs"
    if oa_key:
        return OpenAITTS(
            oa_key,
            args.openai_voice or env.get("OPENAI_TTS_VOICE") or "alloy",
            args.openai_tts_model or env.get("OPENAI_TTS_MODEL") or "gpt-4o-mini-tts",
        ), "openai"
    return None, "none"


# ---------------------------
# Build Anki deck
# ---------------------------


def build_deck(deck_name: str):
    model_id = stable_64bit_int("italify.basic.audio.model.v1")
    deck_id = stable_64bit_int(f"italify.deck.{deck_name}.v1")
    my_model = genanki.Model(
        model_id,
        "Italify: Basic + Audio (v1)",
        fields=[
            {"name": "English"},
            {"name": "Italian"},
            {"name": "Audio"},
        ],
        templates=[
            {
                "name": "Card 1",
                "qfmt": "{{English}}",
                "afmt": '{{FrontSide}}<hr id="answer">{{Italian}}<br><br>{{Audio}}',
            }
        ],
        css="""
.card {
  font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
  font-size: 22px;
  text-align: left;
  color: #222;
  background-color: #fff;
}
hr#answer { margin: 0.8em 0; }
""",
    )
    my_deck = genanki.Deck(deck_id, deck_name)
    return my_model, my_deck


def make_note(model, eng: str, ita: str, audio_tag: str, guid: Optional[str] = None):
    if guid is None:
        # Deterministic GUID from English text (stable across runs)
        guid = sha1_hex(eng)[:16]
    note = genanki.Note(
        model=model,
        fields=[eng, ita, audio_tag],
        guid=guid,
    )
    return note


# ---------------------------
# Main
# ---------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Italify: convert an English text file (one concept per line) into an Anki deck with Italian translations and TTS audio."
    )
    parser.add_argument("input_txt", type=str, help="Path to input plain text (one line per card)")
    parser.add_argument("output_apkg", type=str, help="Path to output .apkg (Anki deck)")
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", "gpt-4o"),
        help="OpenAI model for translation (default: gpt-4o). If you have access to GPT-5, set OPENAI_MODEL=gpt-5.",
    )
    parser.add_argument(
        "--tts-provider",
        choices=["elevenlabs", "openai", "none"],
        default=os.environ.get("TTS_PROVIDER"),
        help="Force TTS provider (default: auto).",
    )
    parser.add_argument(
        "--elevenlabs-voice-id",
        default=os.environ.get("ELEVENLABS_VOICE_ID"),
        help="ElevenLabs voice_id",
    )
    parser.add_argument(
        "--elevenlabs-voice-name",
        default=os.environ.get("ELEVENLABS_VOICE_NAME"),
        help="ElevenLabs voice name (resolved to id)",
    )
    parser.add_argument(
        "--elevenlabs-model",
        default=os.environ.get("ELEVENLABS_MODEL"),
        help="ElevenLabs model_id (default: eleven_multilingual_v2)",
    )
    parser.add_argument(
        "--openai-voice",
        default=os.environ.get("OPENAI_TTS_VOICE", "alloy"),
        help="OpenAI TTS voice (default: alloy)",
    )
    parser.add_argument(
        "--openai-tts-model",
        default=os.environ.get("OPENAI_TTS_MODEL", "gpt-4o-mini-tts"),
        help="OpenAI TTS model (default: gpt-4o-mini-tts)",
    )
    parser.add_argument("--no-audio", action="store_true", help="Skip TTS audio generation")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Debug: skip external API calls (translation echoes input; audio skipped)",
    )
    args = parser.parse_args()

    input_path = Path(args.input_txt)
    output_path = Path(args.output_apkg)
    deck_name = output_path.stem

    if not input_path.exists():
        eprint(f"Input file not found: {input_path}")
        sys.exit(1)

    # Read input
    lines = read_lines(input_path)
    if not lines:
        eprint("No lines to process. Provide a text file with one concept per line.")
        sys.exit(1)

    eprint("Italify - starting")
    eprint(f"- Input: {input_path}")
    eprint(f"- Output deck: {output_path}")
    eprint(f"- Cards to create: {len(lines)}")

    env = dict(os.environ)

    # OpenAI client for translation
    openai_key = env.get("OPENAI_API_KEY")
    if not openai_key and not args.dry_run:
        eprint("OPENAI_API_KEY is not set. Set it to perform translations.")
        sys.exit(1)
    client = OpenAI(api_key=openai_key) if openai_key else None

    # Translate
    translations = translate_lines_openai(lines, client, args.model, dry_run=args.dry_run)

    # TTS selection
    tts_provider = None
    tts_provider_name = "none"
    if not args.no_audio and not args.dry_run:
        tts_provider, tts_provider_name = choose_tts_provider(args, env)
    elif args.no_audio:
        tts_provider_name = "none"

    if args.no_audio:
        eprint("Audio generation disabled (--no-audio).")
    else:
        if tts_provider_name == "elevenlabs":
            eprint("TTS: ElevenLabs (free tier compatible).")
        elif tts_provider_name == "openai":
            eprint("TTS: OpenAI TTS.")
        else:
            eprint(
                "TTS: No provider configured. Set ELEVENLABS_API_KEY (+ optional ELEVENLABS_VOICE_ID) for ElevenLabs, or rely on OpenAI TTS by setting OPENAI_API_KEY."
            )
            eprint("Proceeding without audio.")

    # Prepare temp media dir
    temp_dir = Path(tempfile.mkdtemp(prefix="italify_"))
    media_dir = temp_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)

    # Create audio files (if configured)
    audio_files: List[Path] = []
    audio_tags: List[str] = []

    if tts_provider and not args.no_audio and not args.dry_run:
        for i, ita in enumerate(tqdm(translations, desc="Generating TTS audio", unit="card")):
            base = f"italify_{i + 1:04d}_{sha1_hex(ita)[:8]}"
            filename = sanitize_filename(base) + ".mp3"
            outpath = media_dir / filename

            ok = False
            # Retry a few times
            for attempt in range(3):
                ok = tts_provider.synthesize(ita, outpath)
                if ok:
                    break
                time.sleep(1.5 * (attempt + 1))
            if ok:
                audio_files.append(outpath)
                audio_tags.append(f"[sound:{filename}]")
            else:
                eprint(f"TTS failed for card {i + 1}. Card will have no audio.")
                audio_tags.append("")  # no audio for this card
    else:
        audio_tags = [""] * len(translations)

    # Build deck
    model, deck = build_deck(deck_name)

    for eng, ita, audio_tag in zip(lines, translations, audio_tags):
        note = make_note(model, eng, ita, audio_tag)
        deck.add_note(note)

    package = genanki.Package(deck)
    # Attach media
    package.media_files = [str(p) for p in audio_files]

    eprint("Packaging .apkg ...")
    package.write_to_file(str(output_path))
    eprint(f"Done. Deck generated: {output_path}")
    if tts_provider_name == "none":
        eprint("Note: No audio was added. To include audio:")
        eprint("- Prefer: export ELEVENLABS_API_KEY=... and optionally ELEVENLABS_VOICE_ID=...; or")
        eprint(
            "- Alternative: rely on OpenAI TTS by ensuring OPENAI_API_KEY is set and use --tts-provider openai"
        )
    else:
        eprint(
            "Tip: If audio doesn't play in Anki, ensure 'Play audio automatically' is enabled in Anki preferences."
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        eprint("\nInterrupted.")
        sys.exit(130)

"""
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
"""
