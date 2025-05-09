SOURCES := cable_equation.tex

LATEXMK := latexmk -e '$$clean_ext=q/bbl xdv thm run.xml/' -xelatex -use-make -halt-on-error

all: cable_equation.pdf

cable_equation.pdf: cable_equation.tex # cable.bib
	$(LATEXMK) $<

clean:
	for s in $(SOURCES); do $(LATEXMK) -c "$$s"; done

realclean:
	for s in $(SOURCES); do $(LATEXMK) -C "$$s"; done
