# Dense Representation of the Functional Protein Space with Wasserstein Auto-Encoders

This project aims at applying Wasserstein Auto-Encoders (WAE) to protein function determination, in order to explore the possibility of using the framework as a guiding tool for biologists in the exploration of new protein sequences.

See the [report](report.pdf) and the [poster](poster.pdf) for the project.


## Related Work

We were inspired by the work of Sinai et al., from the Marks Laboratory at Harvard. Instead of using the Variational Auto-Encoder framework, applied the WAE to the same task, hoping to leverage the better results from this novel method.


## Implementation

During this work, we wrote our own implementation of the WAE framework in Pytorch, using Maximum Mean Discrepancy (MMD) as a divergence measure between the marginal and the prior (see section 3.3 in the [report](report.pdf) for more information).

The implementation can be found [here](autoencoders/wasserstein.py).


## Dataset

We used the same dataset as Sinai et al. The folder [`data/`](data/) contains the processed data.


## Training

The folder [`notebooks/`](notebooks/) contains the notebooks that were used to train the models.


## References

See the [reference](reference/) folder for a review of the literature surrounding this project.

For further reference regarding the closely related Variational Auto-Encoder framework, there are nice high-level presentations, for example by [Jaan Altosaar](https://jaan.io/what-is-variational-autoencoder-vae-tutorial/) and [Jeremy Jordan](https://www.jeremyjordan.me/variational-autoencoders/).
