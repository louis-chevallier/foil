start :
	python eulerian.py

start2 :
	python simu.py


start1 :
	python wing.py
	python foil.py

install :
	pip install -q pyfoil
	pip install -q neuralfoil
