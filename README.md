# Does lower numerical precision change what wav2vec2 embeddings tell us?

Storing speech embeddings in fewer bits saves space. This project tests whether it also changes the conclusions drawn from them: are recordings of the same speaker still closer to each other than recordings of different speakers?

`facebook/wav2vec2-base-960h` turns each recording into a 768-dimensional mean-pooled vector, stored as float64, float32, float16 and int8. For each format the pipeline computes cosine distances within and between speakers and compares them through the inter/intra ratio. In my runs the lower precision formats led to the same conclusions as float64 and took less storage.

## Pipeline

Five DVC stages, all settings in `params.yaml`:

| Stage | Script | Output |
|---|---|---|
| prepare | `src/prepare.py` | index of recordings with speaker labels |
| extract | `src/extract.py` | float32 embeddings |
| convert | `src/convert.py` | the four precision formats, with linear quantisation for int8 |
| distances | `src/distances.py` | intra- and inter-speaker cosine distances |
| visualise | `src/visualise.py` | box plots, bar chart, ratio plot |

The corpus has recordings from 19 speaker folders, and the default run samples 50 recordings (`data.sample_size`).

## Run it

```bash
pip install -r requirements.txt
dvc repro
```

## Data

The recordings come from a course speech corpus (19 speaker folders, `.wav` with TextGrid alignments). They are speaker voice data, so they are not published here. Put the corpus under `data/raw/<speaker>/` and run `dvc repro`. The method write-up is in `report.pdf`.

Change `data.sample_size` or `model.name` in `params.yaml` and run `dvc repro` again to rerun only what is affected.
