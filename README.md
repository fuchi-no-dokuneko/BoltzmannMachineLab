# Boltzmann Machine Lab

An offline browser laboratory for comparing a full Boltzmann machine's clamped/free Gibbs phases with an RBM trained using CD-k. The network view exposes stochastic unit states, learned biases, and signed weight magnitude. Activation and update functions are editable and execute only in the current tab.

Training and inference are separate. Inference freezes the current parameters and shows hidden posterior probabilities, visible reconstruction probabilities, sampled states, entropy, and the normalized visible-state distribution. With the standard Boltzmann activation, RBM visible marginals are exact and full-BM marginals are exact for up to 16 total units. Larger models and custom activation functions use a clearly labeled Gibbs estimate.

## References

- Ackley, Hinton, and Sejnowski, *A Learning Algorithm for Boltzmann Machines* (1985): https://onlinelibrary.wiley.com/doi/abs/10.1207/s15516709cog0901_7
- Hinton, *A Practical Guide to Training Restricted Boltzmann Machines*: https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf

This is an educational simulator. Its short Gibbs chains are intentionally interactive and are not a substitute for converged research sampling.

## Run

```bash
./serve-local.sh
```

Open `http://localhost:8085`.

## Credits

See [CREDITS.md](CREDITS.md) for the research sources behind the learning and inference implementation.
