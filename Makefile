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
	cd code && PYTHONPATH=. python3 examples/ch05_features.py
	cd code && PYTHONPATH=. python3 examples/ch06_twoview.py
	cd code && PYTHONPATH=. python3 examples/ch07_vo.py
	cd code && PYTHONPATH=. python3 examples/ch08_ba.py
	cd code && PYTHONPATH=. python3 examples/ch09_posegraph.py
	cd code && PYTHONPATH=. python3 examples/ch10_loops.py
	cd code && PYTHONPATH=.:examples python3 examples/ch11_maps.py
	cd code && PYTHONPATH=.:examples python3 examples/ch12_imu.py

notebooks:
	cd code/notebooks && for nb in *.ipynb; do \
	  python3 -m nbconvert --to notebook --execute --inplace $$nb; done

clean:
	cd course && latexmk -c
