# Third-party checkouts

FireRedASR is cloned here (not vendored in git):

```bash
git clone https://github.com/FireRedTeam/FireRedASR.git third_party/FireRedASR
pip install -r third_party/FireRedASR/requirements.txt
```

On Colab, clone inside `/content/Who_Gets_Filtered_Out` after `git reset --hard origin/main`. Keep `HF_HOME=/content/hf_cache`. A new runtime is safer than mixing with FunASR.

Use a dedicated venv on a cluster; FunASR / qwen-asr / vLLM may conflict.
