# Whisper Agent

## Installation
Download any of the models from the [OpenAI HuggingFace page here](https://huggingface.co/ggerganov/whisper.cpp/tree/main)
```
$ cargo install --git https://github.com/remykarem/whisper-agent whisper-agent
```

## Usage
```
$ whisper-agent <path-to-model>
```

## TODO
- Update whisper-rs to 0.6.0 or latest which use generics
- Upload the package to crates.io for easier installation
- Add built-in model downloading using reqwest or similar
- Add a GitHub Actions Workflow to build release binaries
- Add stream mode to continuously convert speech to words
