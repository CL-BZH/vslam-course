.PHONY: pdf test figures notebooks clean

pdf:
	cd course && latexmk -pdf -interaction=nonstopmode main.tex

test:
	cd code && python3 -m pytest -q

figures:
	cd code && PYTHONPATH=. python3 examples/ch01_linear_slam.py
	cd code && PYTHONPATH=. python3 examples/ch02_banana.py
	cd code && PYTHONPATH=. python3 examples/ch03_cameras.py
	cd code && PYTHONPATH=. python3 examples/ch04_least_squares.py

notebooks:
	cd code/notebooks && for nb in *.ipynb; do \
	  python3 -m nbconvert --to notebook --execute --inplace $$nb; done

clean:
	cd course && latexmk -c
