# Audio-Transcriber
Transcribe audio and video files on your own computer with OpenAI Whisper models. Save the results as text or subtitles. Works offline after the first model download. Sends no audio or text anywhere.

<img width="1429" height="1450" alt="image" src="https://github.com/user-attachments/assets/5fd70da5-2504-4837-b1ba-5806cc085b25" />

## Features
- Transcribes MP3, WAV, M4A, FLAC, OGG, MP4, AVI and MKV files.
- Transcribes a batch of files one after another.
- Saves each result as text (`.txt`), SRT subtitles (`.srt`), WebVTT subtitles (`.vtt`), or any combination.
- Never overwrites an existing result. If `talk.txt` exists, the new result is saved as `talk (2).txt`.
- Downloads each model once and verifies it against its official SHA256 checksum.
- Shows progress within each file. **Cancel** stops within about 30 seconds of audio.
- Portable: settings and models are kept in a folder next to the app.

## Download
Download the latest version from the [Releases page](https://github.com/TariqMK/Audio-Transcriber/releases):

| Platform | File |
|---|---|
| Windows (64-bit) | `Audio-Transcriber-vX.Y.Z-windows-x64.exe` |
| macOS (Apple Silicon) | `Audio-Transcriber-vX.Y.Z-macos-arm64.zip` |
| Linux (64-bit) | `Audio-Transcriber-vX.Y.Z-linux-x64.tar.gz` |

Each download is 300 MB to 800 MB because it includes PyTorch and Whisper.

**Note:** The builds are not code-signed.
- macOS: the first time you open the app, macOS blocks it. Open **System Settings > Privacy & Security** and select **Open Anyway**.
- Windows: if SmartScreen shows a warning, select **More info**, then **Run anyway**.

## Install ffmpeg
The app needs ffmpeg to read audio. Install it once:
- macOS: `brew install ffmpeg` (needs [Homebrew](https://brew.sh))
- Windows: `winget install Gyan.FFmpeg`
- Linux (Debian/Ubuntu): `sudo apt install ffmpeg`

If ffmpeg is missing, the app tells you when you select **Start Transcription**.

## Set up the app
1. Put the app in a folder you can write to, for example a new folder in Documents.
   **Note:** The app creates its data folder, `Audio-Transcriber-data`, next to itself. Do not run it from a read-only folder such as `Program Files`.
   **Note (macOS):** Unzip the download and move `Audio-Transcriber.app` out of Downloads before you open it.
2. Open the app.

## How to use
1. Under **Model Selection**, choose a model.
   **Note:** Larger models are more accurate, but slower, and need more memory. The sizes are 75 MB (tiny) to 3 GB (large-v3).
2. Select **Download & Load Model**. The first time, the app downloads the model into the **Models Folder**. After that, it uses the saved copy.
   **Note:** If a model file is incomplete or damaged, the app asks before it downloads a new copy. The damaged file is replaced only after the new download is verified.
3. Under **Output Settings**, select **Browse** next to **Output Directory** and choose where to save the results.
4. Under **Save As**, select one or more formats.
5. Select **Select File(s)** and choose the files to transcribe. The app skips files that are already in the list.
6. Select **Start Transcription**.
7. To stop, select **Cancel**. Files that are not finished are not saved.

The app names each result after its source file, for example `interview.mp3` gives `interview.txt`. Errors are written to a log in the **Error Log Directory**. To use other folders for models or logs, select **Browse** next to that folder.

## Check for updates
1. Select the version number next to the title, for example **v1.2.0**.
2. Read the result. If a newer version exists, select **Open release page** to download it.

The app checks only when you select the version number. It never checks on its own.

## Privacy
- Transcription runs fully on your computer. Audio and transcripts never leave it.
- No telemetry, analytics or tracking.
- With a model downloaded, the app works with no internet connection.
- Settings, models and error logs are stored in the `Audio-Transcriber-data` folder next to the app. Move the app and that folder together to keep everything.
- Transcripts are saved only in the folder you choose.
- The app makes only two network connections, and only when you ask:
  - When you select **Download & Load Model** and that model is not yet downloaded, the app downloads it from OpenAI's model server (`openaipublic.azureedge.net`). It sends nothing except the request for the file.
  - When you select the version number, the app asks GitHub (api.github.com) for the latest release. It sends nothing else and nothing is downloaded.

## Run from source
You need Python 3.9 or later and ffmpeg.

1. Clone this repository and open a terminal in its folder.
2. Create a virtual environment: `python3 -m venv .venv`
3. Activate it:
   - macOS and Linux: `source .venv/bin/activate`
   - Windows: `.venv\Scripts\activate`
4. Install the app and its dependencies: `pip install -e ".[dev]"`
5. Start the app: `python -m audio_transcriber`

A source run shows the version `v0.0.0.dev0` and keeps its data in `Audio-Transcriber-data` in the project folder.

**Note:** Settings from versions before the data folder (`~/.whisper_gui_settings.json`) are copied over on first start. The old file is not changed or deleted.

## Run the tests
1. Install the development dependencies (step 4 above).
2. Run: `python -m pytest`

The tests use a fake Whisper model. They do not need PyTorch, a model download or a network connection.

## Creating a Release
Releases are built by the **Release** workflow in GitHub Actions.

1. Open the **Actions** tab on GitHub and select **Release**.
2. Select **Run workflow** on the `main` branch.
3. Choose the **Release type**:
   - `draft` creates a draft release that only you can see. Publish it on GitHub when it is ready.
   - `public` publishes the release immediately.
4. Choose the **Version increment**: `patch` (1.2.3 → 1.2.4), `minor` (1.2.3 → 1.3.0) or `major` (1.2.3 → 2.0.0).
5. Select **Run workflow**.

The workflow runs the tests first. It then builds Windows, macOS and Linux versions and checks that each build loads Whisper and its data files. It creates the tag and the release only when every build succeeds.

The next version comes from the highest existing `vX.Y.Z` tag or release, including drafts. The first release is `v0.0.1` for a patch. The version is written into the app on the build machine only. The workflow never commits to any branch. No secrets need to be set up.
