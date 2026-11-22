# User guide

Start with the input data format and the recommended workflow, then the guides to choosing K and the density
threshold. The stage reference pages after that cover each stage's inputs, parameters and outputs; they are built
from the README in each stage's folder, so they always match the code.

```{toctree}
:maxdepth: 1
:caption: Before you start

data_format
workflow
```

```{toctree}
:maxdepth: 1
:caption: Choosing parameters

choosing_k
density_threshold
```

```{toctree}
:maxdepth: 1
:caption: Stage 1 · Inference

inference
sk_cnmf
torch_cnmf
```

```{toctree}
:maxdepth: 1
:caption: Stage 2 · Evaluation

evaluation
metrics
calibration
calibration_utest
calibration_crt
calibration_matched_de
```

```{toctree}
:maxdepth: 1
:caption: Stage 3 · Interpretation

interpretation
plotting
summarization
annotation
```

```{toctree}
:maxdepth: 1
:caption: Other

hpc
```
