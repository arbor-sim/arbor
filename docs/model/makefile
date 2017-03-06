report.pdf : *.tex images.dir bib
	pdflatex report.tex

images.dir:
	make -C images

force : report.pdf
	pdflatex report.tex

bib : *.tex bibliography.bib
	pdflatex report.tex
	bibtex report
	pdflatex report.tex
	pdflatex report.tex

clean :
	rm -f *.pdf
	rm -f *.log
	rm -f *.out
	rm -f *.aux
	rm -f *.toc
