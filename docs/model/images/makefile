.PHONY: all

sources := cable.tex soma.tex

all: $(patsubst %.tex,%.pdf,$(sources))

LATEXMK := latexmk -pdf -halt-on-error

%.pdf: %.tex
	$(LATEXMK) $<

clean:
	for s in $(sources); do $(LATEXMK) -C "$$s"; done

