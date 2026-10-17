---
title: Paper, software and method citations
description: Full citation and BibTeX for the 2022 Laplace HypoPINN article, with software provenance, approximation references and MIT license scope.
---

# Cite Laplace HypoPINN

The original Laplace HypoPINN article introduces physics-informed neural-network
hypocenter localization and its predictive uncertainty:

Muhammad Izzatullah, Isa Eren Yildirim, Umair Bin Waheed and Tariq Alkhalifah
(2022). **Laplace HypoPINN: physics-informed neural network for hypocenter
localization and its predictive uncertainty.** *Machine Learning: Science and
Technology*, **3**(4), 045001.
[https://doi.org/10.1088/2632-2153/ac94b3](https://doi.org/10.1088/2632-2153/ac94b3).

```bibtex
@article{Izzatullah2022LaplaceHypoPINN,
  author = {Izzatullah, Muhammad and Yildirim, Isa Eren and
            Waheed, Umair Bin and Alkhalifah, Tariq},
  title = {{Laplace HypoPINN}: physics-informed neural network for
           hypocenter localization and its predictive uncertainty},
  journal = {Machine Learning: Science and Technology},
  volume = {3},
  number = {4},
  pages = {045001},
  year = {2022},
  doi = {10.1088/2632-2153/ac94b3},
  url = {https://doi.org/10.1088/2632-2153/ac94b3}
}
```

This reference matches the repository's
[`CITATION.cff`](https://github.com/izzatum/laplace-hypopinn/blob/main/CITATION.cff),
which also provides machine-readable software citation metadata. The maintained
implementation is available at
[`github.com/izzatum/laplace-hypopinn`](https://github.com/izzatum/laplace-hypopinn).
Record the commit or release used in your experiments alongside the paper
citation so readers can identify the executed software.

## Added approximation methods

Cite the relevant method sources when using the maintained research extensions.
The [method reference](methods.md) links the KFAC, EKFAC-style EFB and VaLLA
publications and describes the archived INF source. It specifies where the
PINN residual and coordinate-derivative adaptations differ from those sources.
The 2022 article alone does not describe every method in the current package.

The [reproduction guide](reproducibility.md) distinguishes the original notebook
conventions, paper differences and maintained alternatives. The
[performance guide](performance.md) states which numerical comparisons have been
completed; cite or report those results only within their recorded settings.

## License scope

The maintained package, documentation and new tutorials are licensed under
[MIT](https://github.com/izzatum/laplace-hypopinn/blob/main/LICENSE). Historical
archive material, the bundled Otway dataset and third-party dependencies have
separate terms, documented in
[`THIRD_PARTY_NOTICES.md`](https://github.com/izzatum/laplace-hypopinn/blob/main/THIRD_PARTY_NOTICES.md).
The article's CC BY 4.0 terms apply separately to the publication.
