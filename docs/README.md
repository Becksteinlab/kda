# Compiling Kinetic Diagram Analysis's Documentation

The docs for this project are built with [Sphinx](http://www.sphinx-doc.org/en/master/).
To compile the docs, first ensure that Sphinx and the ReadTheDocs theme are installed.


```bash
pip install sphinx sphinx_rtd_theme
```


Once installed, you can use the `Makefile` in this directory to compile static HTML pages by
```bash
make html
```

The compiled docs will be in the `_build` directory and can be viewed by opening `index.html` (which may itself 
be inside a directory called `html/` depending on what version of Sphinx is installed).

# BibTex Citations

Citations are handled using the `sphinxcontrib.bibtex` extension. References are 
stored in `docs/references.bib` and can be cited using `:footcite:`. Citations for
each page can be added by including the sphinx directive `.. footbibliography::`.
A good explanation of usage can be found [here](https://chiplicity.readthedocs.io/en/latest/Using_Sphinx/UsingBibTeXCitationsInSphinx.html#id6).
