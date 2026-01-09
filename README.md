# Fader Networks — Image Attribute Manipulation (MLA 2025/2026)

Re-implementation and reproduction of **Fader Networks** (Lample et al., 2018): an **encoder–decoder** model trained with an **adversarial discriminator in latent space** to learn a representation *invariant* to chosen attributes, enabling **continuous attribute manipulation** (“sliding attributes”) while preserving the content/identity of the image.

This project was developed in the context of the **Advanced Machine Learning** course (Sorbonne Université). We adopt a progressive strategy:
- **Colored-MNIST**: proof-of-concept for clean disentanglement (binary color attribute).
- **Oxford Flowers-102**: transition to natural images (64×64), attribute derived from captions (e.g., “pink”).
- **CelebA**: attempt to reproduce paper-style attribute manipulation at 256×256.

> Key idea: the discriminator predicts attributes from the latent `z`, while the encoder learns to **hide** attribute information in `z`, producing an attribute-invariant latent space. The decoder receives `(z, y)` to reconstruct / manipulate images.

Because the datasets are large, they are **NOT stored in the Git repository**.  
Instead, they we provide them externally as **split compressed files** that you download, rebuild into a single archive, then extract locally.

## Repository layout

```text

FADER_NETWORKS/
├── datasets/
│   ├── celeb_a/                 # extracted dataset folder
│   ├── flowers/                 # extracted dataset folder
│   ├── MNIST/                   # extracted dataset folder
│   └── compressed_data/
│       ├── datasets.tar.gz          # main archive (LFS)
│       ├── datasets.tar.gz_part_aa  # optional split parts (LFS)
│       ├── datasets.tar.gz_part_ab
│       └── datasets.tar.gz_part_ac
├── celeb_a.ipynb
├── flowers.ipynb
├── mnist.ipynb
├── Fader_2018.pdf
└── README.md
