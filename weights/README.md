# Trained generator checkpoint

The synthetic arm of the pipeline needs the trained generator released with the
paper. Place it in this directory and pass its path to `--weights`:

```
weights/generator.pt
```

The checkpoint holds the state dictionary of `ImprovedGeneratorNoTanh`
(`src/generator.py`), trained on EPHNOGRAM 8 s phonocardiogram-ECG pairs at 8 kHz.
The loader accepts either a bare state dictionary or a training checkpoint that
carries one under `generator_state_dict` or `state_dict`, and loads with
`strict=True`, so a file that does not match the architecture raises immediately.

The checkpoint is distributed with the article's supplementary materials and from
the corresponding author. The fixed-window control arm (`--boundaries fixed`) runs
without it.
