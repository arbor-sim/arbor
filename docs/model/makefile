images := images/cable.pdf images/soma.pdf

LATEXMK := latexmk -e '$$clean_ext=q/bbl run.xml/' -pdf -use-make -halt-on-error

report.pdf : report.tex appendix.tex formulation.tex symbols.tex SelfArx.cls bibliography.bib ${images}
	$(LATEXMK) $<

images/%.pdf: images/%.tex
	make -C images $(patsubst images/%,%,$@)

clean:
	$(LATEXMK) -C report.tex
	make -C images clean    
