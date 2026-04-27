<h1 align="center">KAME</h1>

<p align="center">
  <strong>KAME: TANDEM ARCHITECTURE FOR ENHANCING KNOWLEDGE IN REAL-TIME SPEECH-TO-SPEECH CONVERSATIONAL AI</strong>
</p>

<p align="center">
  <a href=".github/workflows/precommit.yml"><img alt="Checks" src="https://img.shields.io/badge/checks-passing-brightgreen"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-%3E%3D3.10-blue">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
  <a href="https://docs.astral.sh/ruff/"><img alt="ruff" src="https://img.shields.io/badge/code%20style-ruff-informational"></a>
</p>

<p align="center">
  <a href="https://github.com/SakanaAI/kame_finetune">KAME Finetuning</a> ·
  <a href="https://arxiv.org/abs/2510.02327">Paper</a> ·
  Blog post: coming soon
</p>

KAME is a spoken dialogue system built on top of the
[Kyutai Moshi](https://github.com/kyutai-labs/moshi) codebase.
This repository keeps the Python inference stack needed for:

- running KAME's oracle-guided dialogue server with a web UI
- loading KAME-compatible `kame` modules from `kame_finetune`

The public-facing focus of this repository is the Python inference path around
`kame.server_oracle`, while keeping the generic `kame.server` flow available
for compatibility.

<p align="center">
  <img src="docs/assets/kame-overview.gif" alt="KAME oracle-guided spoken dialogue demo in the browser UI" width="900">
</p>

<p align="center">
  <em>KAME running oracle-guided spoken dialogue with live browser interaction.</em>
</p>

## What KAME Adds

Compared with the upstream Moshi repository, KAME adds and maintains the
oracle-guided dialogue path used for our experiments and demos.
The primary entrypoint is:

```bash
python -m kame.server_oracle --help
```

or, after installing the package in editable mode:

```bash
kame-server-oracle --help
```

This server provides the KAME-specific inference path and serves a browser UI.
If `--static` is not provided, the server can fetch static assets automatically.
For compatibility, the generic Python server is also retained:

```bash
python -m kame.server --help
```

## Runtime Notes

- `kame.server_oracle` requires `OPENAI_API_KEY`.
- ASR is enabled by default and uses OpenAI Realtime transcription
  (`gpt-4o-mini-transcribe`). Use `--asr-provider google` to use Google Cloud
  Speech-to-Text instead.
- The current oracle-guided server path is configured for English dialogue and ASR.
- If `--static` is omitted, the browser UI assets are fetched automatically at startup.
- `kame.server_oracle` sends conversation text to OpenAI Chat Completions.
- If ASR is enabled, `kame.server_oracle` sends audio to the selected ASR provider.
- `kame.server_oracle` currently supports only a single active WebSocket session at a time; concurrent sessions are rejected with `503 Server busy`.
- Plaintext local session logs are disabled by default. Enable them explicitly with `--log-dir` or `MOSHI_LOG_DIR` if you want to persist transcripts and token streams locally.

## Repository Layout

The parts of this repository that matter for KAME are:

- [`src/kame/`](src/kame/): installable KAME Python package
- [`src/kame/server_oracle.py`](src/kame/server_oracle.py): oracle-guided server entrypoint
- [`src/kame/server.py`](src/kame/server.py): generic non-oracle server retained for compatibility
- [`src/kame/models/`](src/kame/models/): language model and checkpoint loading code used by `kame_finetune`

The published distribution name is `kame-model`, while the Python import
namespace is `kame`. This repository now uses a standard root project layout
with the package source under [`src/kame/`](src/kame/).

## Typical Usage

### Run from the Hugging Face Checkpoint

The public checkpoint can be loaded directly from Hugging Face with
`--hf-repo`. The package distribution name is `kame-model`, while the Python
module name is `kame`.

```bash
uv init --bare --python 3.12
uv add "kame-model @ git+https://github.com/SakanaAI/kame.git@main"

export OPENAI_API_KEY=...

uv run python -m kame.server_oracle \
  --hf-repo SakanaAI/kame \
  --host 0.0.0.0 \
  --port 8998 \
  --device cuda
```

Then open `http://localhost:8998`.

`kame-model` is not published on PyPI yet, so the example above installs it
directly from GitHub. For reproducible runs, pin a release tag or commit instead
of installing from `main`.

Notes:

- Python `>=3.10` is supported; the command above uses Python 3.12 because it is
  the version used for verification.
- `OPENAI_API_KEY` is required by `kame.server_oracle` and is used for both the
  oracle LLM stream and the default OpenAI ASR provider.
- To use Google Cloud Speech-to-Text instead of OpenAI ASR, set up a Google
  Cloud project for
  [Speech-to-Text](https://cloud.google.com/speech-to-text/docs/setup) and
  configure
  [Application Default Credentials](https://cloud.google.com/docs/authentication/set-up-adc-on-premises)
  with `GOOGLE_APPLICATION_CREDENTIALS`, then pass `--asr-provider google`.
- For local smoke tests without ASR, pass `--no-enable-asr`.
  This skips ASR and does not exercise the full oracle-guided spoken-dialogue path.
- `--config-path`, `--moshi-weight`, `--mimi-weight`, and `--tokenizer` are not
  needed for the public Hugging Face checkpoint in the usual case.
- `config.json` in the Hugging Face repo resolves the model weights, Mimi
  checkpoint, tokenizer, and optional generation settings.

### Local Development

```bash
pip install -e .
python -m kame.server_oracle --help
```

`kame_finetune` can then depend on this repository directly from the repo root,
for example via a local editable path dependency.

## Scope

This repository is intentionally narrower than the original Moshi release.
The main supported workflow is:

1. install the Python package from the repository root
2. run `server_oracle.py` for oracle-guided interactive inference, or `server.py` for the generic server path
3. use the same Python package as the `kame-model` dependency from `kame_finetune`

## License

The `kame-model` Python package is distributed under the MIT License.
This repository is derived from the Kyutai Moshi codebase and retains the
relevant upstream license files and notices. Additional inherited notices,
including [`LICENSE.audiocraft`](LICENSE.audiocraft), are kept at the project
root. Model weights and datasets, when distributed separately, may be subject to
different license terms.

## Attribution

KAME is derived from the
[Kyutai Moshi repository](https://github.com/kyutai-labs/moshi).
We retain the original license files and attribution for the inherited codebase,
and extend the Python inference stack with KAME-specific functionality.

Please keep the existing license files in this repository, including:

- [`LICENSE`](LICENSE)
- [`LICENSE.audiocraft`](LICENSE.audiocraft)
- [`LICENSE-MIT`](LICENSE-MIT)

## Citation

If you use KAME in your research, please cite:

```bibtex
@article{kuroki2025kame,
  title={KAME: Tandem Architecture for Enhancing Knowledge in Real-Time Speech-to-Speech Conversational AI},
  author={Kuroki, So and Kubo, Yotaro and Akiba, Takuya and Tang, Yujin},
  journal={arXiv preprint arXiv:2510.02327},
  year={2025}
}
```
