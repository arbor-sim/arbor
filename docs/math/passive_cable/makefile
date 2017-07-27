SOURCES := cable_computation.tex

LATEXMK := latexmk -e '$$clean_ext=q/bbl run.xml/' -xelatex -use-make -halt-on-error

all: cable_computation.pdf

cable_computation.pdf: cable_computation.tex cable.bib
	$(LATEXMK) $<

clean:
	for s in $(SOURCES); do $(LATEXMK) -c "$$s"; done

realclean:
	for s in $(SOURCES); do $(LATEXMK) -C "$$s"; done
