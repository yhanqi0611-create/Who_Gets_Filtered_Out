# Third-party checkouts

FireRedASR is cloned here (not vendored in git):

```bash
git clone https://github.com/FireRedTeam/FireRedASR.git third_party/FireRedASR
pip install -r third_party/FireRedASR/requirements.txt
```

Use a dedicated venv; its dependencies may conflict with funasr / qwen-asr / vLLM.
