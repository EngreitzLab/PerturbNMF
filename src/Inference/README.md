# Inference

## Difference Between sk-cNMF and torch-cNMF

sk-cNMF uses coordinate descent, which requires element-wise updates that cannot be parallelized on GPU.
Instead of updating one element at a time, torch-cNMF with the HALS solver updates a column of elements at a time.
Through the benchmarking process, updating a column of elements was found to lead to lower stability and lower biological metrics using the same hyperparameters.
However, with lower tolerance in torch-cNMF HALS, torch-cNMF is able to achieve very similar results. Currently, we are not sure how this difference will change the downstream analysis.

Therefore, currently, we think torch-cNMF can be used for K selection, with the following hyperparameters:
- Tolerance: 1e-7
- NMF replicates: 20

and sk-cNMF can be used for running the final K, with the following hyperparameters:
- Tolerance: 1e-4
- NMF replicates: 100

Benchmarking result: https://docs.google.com/presentation/d/1Z25ew7xrnhXD_eQx7e7eg6vtHx_T4uVD/edit?usp=sharing&ouid=103348313942131245812&rtpof=true&sd=true


## Recommended Steps for K Selection 

1. Select as many genes as possible as highly variable genes (HVGs), within time and memory constraints, to increase enriched term numbers.
2. For larger datasets, use a lower convergence tolerance and increase the number of iterations to maximize solution stability.
3. Perform a broad sweep across a wide range of K values to inspect the overall K-selection trends in basic stability and biological metrics.
4. If necessary, conduct more targeted runs with denser sampling around the best-performing K range identified in the initial sweep.
5. Examine perturbation calibration plots to determine the most appropriate p-value estimation method.
6. Select the optimal K by integrating all available information, including stability, biological term metrics and gene annotations.
7. Test different density thresholds on the selected K to determine the best filter.
8. Finally, generate a comprehensive set of downstream analysis and visualization plots for the selected K value.
