# Figure entry points

Article-specific figure-generation commands will live here.

Numerical models should remain importable and testable under `src/blog_reproducibility/`. These scripts should be thin entry points that load the model, render the required figure, and write generated output to `build/figures/`.

Publication-ready copies of rendered figures remain in the website repository.
