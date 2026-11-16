.PHONY: pdf test figures clean

pdf:
	cd course && latexmk -pdf -interaction=nonstopmode main.tex

test:
	cd code && python3 -m pytest -q

figures:
	cd code && PYTHONPATH=. python3 examples/ch01_linear_slam.py
	cd code && PYTHONPATH=. python3 examples/ch02_banana.py

clean:
	cd course && latexmk -c
