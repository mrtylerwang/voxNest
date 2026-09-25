# voxNest

> Local AI voice studio for Apple Silicon Macs — open source speech-to-text and text-to-speech that runs entirely offline.

<div align="center">

[English](README.md) · [中文](README.zh.md)

<br>

<a href="https://github.com/mrtylerwang/voxnest/actions"><img src="https://img.shields.io/github/actions/workflow/status/mrtylerwang/voxnest/build.yml?branch=main&label=Build" alt="Build" /></a>
<a href="https://github.com/mrtylerwang/voxnest/releases/latest"><img src="https://img.shields.io/github/v/release/mrtylerwang/voxnest?label=Release" alt="Latest Release" /></a>
<a href="https://github.com/mrtylerwang/voxnest/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow" alt="License" /></a>

</div>

<div align="center">
  <img src="assets/voxNest.gif" width="720" alt="voxNest demo" />
</div>

voxNest brings on-device AI speech to Apple Silicon Macs. Everything runs locally using the Apple MLX framework — no audio uploads, no account, no telemetry, no subscription. Ships with an embedded Python runtime and ffmpeg; users install nothing extra.

Supported on **all Apple Silicon Macs**, macOS 15 Sequoia or later. Intel Macs, Windows, and Linux are not supported.

---

## Install

| Region | Download |
|---|---|
| International | [`voxNest_aarch64.dmg`](https://github.com/mrtylerwang/voxnest/releases/latest/download/voxNest_aarch64.dmg) |
| Mainland China mirror | [`voxNest_aarch64.dmg (gh-proxy)`](https://gh-proxy.com/https://github.com/mrtylerwang/voxnest/releases/latest/download/voxNest_aarch64.dmg) |
| All releases | [GitHub Releases](https://github.com/mrtylerwang/voxnest/releases) |

1. Double-click the `.dmg`, drag **voxNest** into Applications
2. Launch — backend starts automatically. On first run, an embedded Python runtime installs inside the app's data directory
3. From the Model Manager, download the ASR and TTS models you need (progress display, resumable)

### macOS warning: "voxNest is damaged and can't be opened"

On first launch, macOS Gatekeeper may block the app with a "damaged" warning because the `.dmg` was downloaded from the internet. This is a standard quarantine attribute, not an actual corruption.

**Fix** — open Terminal and run:

```bash
sudo xattr -r -d com.apple.quarantine /Applications/voxNest.app
```

Enter your Mac password when prompted, then launch voxNest again. The warning will not reappear.

## Speech-to-Text (ASR)

Drop an audio file, get three views: **transcript** (plain text), **timeline** (per-segment timestamps), **SRT** (subtitle format).

| Model | Size | Languages | Notes |
|---|---|---|---|
| Qwen3-ASR-1.7B-8bit | ~1.8 GB | 52 languages, auto-detection | Flagship accuracy, native timestamps, built-in punctuation |
| Fun-ASR-Nano-0.8B | ~250 MB | Chinese + 7 dialects, lyrics | Lightweight, fast |

## Text-to-Speech (TTS) — Three Modes

All TTS variants depend on a shared tokenizer model (≈ 0.65 GB), downloaded automatically with the first TTS model.

| Mode | Input | Model | Notes |
|---|---|---|---|
| **Base (Voice Clone)** | Source audio | 12Hz-1.7B-Base-8bit | Reference text auto-transcribed by the built-in ASR engine |
| **CustomVoice (Preset)** | Voice selection | 12Hz-1.7B-CustomVoice-8bit | 9 built-in preset voices; reference audio not accepted |
| **VoiceDesign** | 7-dimension description | 12Hz-1.7B-VoiceDesign-8bit | Gender + age + pitch + rate + emotion + timbre + use case |

Output is post-processed — loudness normalization, crossfades at chunk boundaries, tail-preserving silence trimming.

---

## FAQ

**Which Macs does voxNest support?**
All Apple Silicon Macs (any M-series chip), macOS 15 Sequoia or later. Intel Macs, Windows, and Linux are not supported — MLX requires Apple Silicon.

**Does voxNest require internet?**
Only for the initial model download. After that, every feature runs entirely offline.

**How do I choose between the three TTS modes?**
Use **Base** to clone a voice from a reference clip. Use **CustomVoice** for one of 9 built-in preset voices. Use **VoiceDesign** to describe a voice in natural language.

**Is voxNest free?**
voxNest itself is MIT-licensed and free for personal and commercial use. Model weights follow their respective licenses — check ModelScope or HuggingFace for details.

**macOS says "voxNest is damaged and can't be opened"**
This is the Gatekeeper quarantine attribute, triggered because the `.dmg` was downloaded. Fix: `sudo xattr -r -d com.apple.quarantine /Applications/voxNest.app` — see the Install section above for details.

---

## Links

| | |
|---|---|
| Product site | [https://voxnest.cnxc.me/](https://voxnest.cnxc.me/) |
| Product site (Chinese) | [https://voxnest.cnxc.me/zh/](https://voxnest.cnxc.me/zh/) |
| Downloads | [GitHub Releases](https://github.com/mrtylerwang/voxnest/releases/latest) |

## Contributing

Issues and PRs welcome. Please open an issue first for larger changes. Current scope is intentionally limited — Apple Silicon macOS only, MLX inference only.

## License

[MIT](LICENSE). Made with ♥ in Cupertino & Chengdu.
