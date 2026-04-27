# COOM Benchmark: HACE + PackNet

This document explains how to run the **PackNet + HACE** method on the **COOM benchmark**.

## Setup

Clone the COOM repository:

```bash
git clone <COOM_REPO_URL>
cd COOM
```

Copy the HACE + PackNet implementation into COOM's continual learning methods directory:

```bash
cp packnet_hace.py COOM/CL/methods/
```

The expected file path is:

```text
COOM/CL/methods/packnet_hace.py
```

## Training

Run the continual learning experiment with `packnet_hace` as the CL method:

```bash
python run_cl.py --cl_method packnet_hace
```

## Testing / Evaluation

After training, set the checkpoint path for the trained model you want to evaluate.

`MODEL_PATH` should be relative to:

```text
${REPO_ROOT}/checkpoints/
```

Set `MODEL_PATH` using the checkpoint directory generated during training:

```bash
MODEL_PATH="hace/REPLACE_WITH_YOUR_TIMESTAMP_ContinualLearningEnv"
```

Then run evaluation with the saved checkpoint:

```bash
python run_cl.py \
    --cl_method packnet_hace \
    --test \
    --model_path "$MODEL_PATH"
```

## Notes

- Replace `<COOM_REPO_URL>` with the actual COOM repository URL.
- Replace `REPLACE_WITH_YOUR_TIMESTAMP_ContinualLearningEnv` with the actual checkpoint folder name.
- Use `--test` only when evaluating a trained checkpoint.
- `MODEL_PATH` should not include `${REPO_ROOT}/checkpoints/`; it should only include the relative checkpoint path.