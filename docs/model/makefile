report.pdf : *.tex
	pdflatex report.tex

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
