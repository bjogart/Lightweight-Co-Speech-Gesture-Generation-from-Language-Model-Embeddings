# Lightweight Co-Speech Gesture Generation from Language Model Embeddings

This repository contains data and code of the master's thesis of the same name
as a supplement to the main report and stimulate further research.

## Perceptual Study Data

Data gathered as part of our perceptual study is published, in anonymized form,
in the [votes/](./votes/) directory. The data is structured to resemble the
format used by
the[GENEA Gesture-Generation Leaderboard](https://genea-workshop.github.io/leaderboard);
the files are compatible with their
[statistical analysis scripts](https://github.com/genea-workshop/leaderboard-statistical-analysis),
which constituted a part of our own evaluation as well.

## Code

This project was developed in Python. We provide four user-facing scripts, with
auxiliary code in the [scripts](./scripts/) directory.

Code is published as four user-facing scripts:

- **`train.py`** trains a latent pose predictor, a small classifier model which
  uses text embeddings from a pre-trained LLM to predict gesture. Run the
  following command to train a pose predictor with default hyperparameters.

  ```bash
  $ python train.py predictor_weights
  ```

  Checkpoints and training metadata will be saved to `./predictor_weights/`.
- **`pipeline.py`** generates and renders gesture for a string of text.

  ```bash
  $ python pipeline.py weights/predictor--epoch08.pth --no-lower --subs "'The quick brown fox jumps over the lazy dog.'Only a handful of people have ever witnessed such a scene in reality, probably, but the number of people that have seen it in type is massive, without a doubt. Did you know?---I was once among those lucky few that saw the real thing! I had a camera with me too. I took as many as ten photographs...but I lost my only copies when my computer crashed."
  ```

  FFmpeg installation is required to render video.
- **`metrics.py`** generates gestures for pose predictor weights, and compares
  them to ground-truth motion from the
  [BEAT2](https://huggingface.co/datasets/H-Liu1997/BEAT2) test split.

  ```bash
  $ python metrics.py weights/predictor--epoch08.pth --stride 2
  ```

  Computed values are saved to `objmetrics.json` by default, or another path if
  specified with `--output`.
- **`bench.py`** benchmarks a pose predictor, or EMAGE, a gesture-generation
  model used as a comparison.

  ```bash
  $ python bench.py predictor weights/predictor--epoch08.pth --stride 2
  $ python bench.py emage
  ```

  By default, results are saved to `bench_results.json`. A different path may be
  specified with `--output`.

[weights/predictor--epoch08.pth](./weights/predictor--epoch08.pth) contains the
pose predictor weights that was evaluated in the thesis. Command-line arguments
for all scripts default to the hyperparameters of this particular model. To
train or test a different model, use the relevant command-line flags for each
script to specify the necessary values. See each script's `--help` output for
available options.

### Setup

Code in this project was developed with [uv](https://github.com/astral-sh/uv).
It should be sufficient to run the following commands to initialize a virtual
environment and install required packages:

```bash
# Create a new virtual environment.
$ uv venv
# Activate the new virtual environment in the current shell instance.
$ .venv/scripts/activate
# Install packages listed in `pyproject.toml`.
$ uv sync
```

Running [prepare.py](./prepare.py) handles subsequent setup.

| Operation                                                                    | Artifact               | Size on Disk | Resumable |
| :--------------------------------------------------------------------------- | ---------------------- | -----------: | :-------: |
| Clone [PantoMatrix](https://github.com/PantoMatrix/PantoMatrix) repository   | `./PantoMatrix/`       |     604.7 MB |           |
| Create [data/](./data/) directory                                            | `./data/`              |       0.0 GB | &#10004;  |
| Download [EMAGE evaltools](https://huggingface.co/H-Liu1997/emage_evaltools) | `./emage_evaltools/`   |     184.9 MB |           |
| Download BEAT2                                                               | `./data/beat2/`        |      20.3 GB |           |
| Precompute foot contact positions                                            | `./data/foot_contact/` |        <1 MB | &#10004;  |
| Precompute text embeddings                                                   | `./data/text_embeds/`  |       2.8 GB | &#10004;  |
| Precompute codebook indices                                                  | `./data/pose_idxs/`    |      12.8 MB | &#10004;  |
| Pack related data together to minimize disk IO                               | `./data/pack/`         |       6.1 GB | &#10004;  |

Some operations can take substantial amounts of time. Operations marked as
resumable can be interrupted; if so, running `python prepare.py` again will
resume the operation where it stopped. If an operations that cannot be resumed
is interrupted, its associated artifact should be erased, and
`python prepare.py` should be run again to re-create the artifact in full.

Note that these artifacts take up a large amount of disk space: 38.9 GB when
including EMAGE and the LLM, which are downloaded when they are used in a
script. Once `prepare.py` runs to conclusion succesfully, foot contact
positions, text embeddings, and codebook index directory are no longer necessary
to any script, and can be deleted safely.

### References

Some scripts were adapted from prior work.

- [scripts/metrics_genea.py](./scripts/metrics_genea.py) was developed from
  [scripts for the GENEA Leaderboard](https://github.com/GENEALeaderboard/objective_metric/blob/02ba223acd8df61c33ab0acacab1331d6dbbe0cd).
- [scripts/render.py](./scripts/render.py) was inspired by a similar script in
  the
  [PantoMatrix repository](https://github.com/PantoMatrix/PantoMatrix/blob/c7356f35f8e39e469e510ccd1bf37e44adf8ec0e/emage_utils/fast_render.py).
