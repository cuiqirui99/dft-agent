# Model context checks

Three paired inputs use the same goal, complete structure, instructions and response schema:

- `long-revision`: eight user replies, including site moments and an early oxygen +U exclusion.
- `large-geometry`: all 256 sites, encoded as objects or a compact table.
- `retrieved-advisory`: verified saved NiO cases. These short moment arrays remain complete.

The dialogue is a constructed test case. Saved-run values are retained evidence, not new calculations. The fixtures contain no credentials or cluster paths.

Check the fixtures without calling a model:

```bash
python validation/token-use/run_benchmark.py
```

Run the six requests with your configured Codex login:

```bash
python validation/token-use/run_benchmark.py --run --provider codex --model YOUR_MODEL --output token-results.json
```

Use `--case long-revision` for one pair. API providers are `responses` and `chat_completions`; configure authentication through the environment.

The runner checks tasks, parameters and exclusions, and records provider-reported input, output and cached tokens separately. Unknown counts stay unknown. One call per input does not establish average savings or latency; byte counts are not token counts.

The large advisory-vector branch has a synthetic offline test in `tests/test_prompt_context.py`. No synthetic vectors are presented as verified past calculations in these fixtures.
