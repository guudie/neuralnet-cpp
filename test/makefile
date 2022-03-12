all: test

test: test.o
	g++ test.o -o test

test.o: test.cpp
	g++ test.cpp -I../src/ -c -O2 -o test.o 

clean:
	rm test test.o

force:
	g++ test.cpp -I../src/ -c -O2 -o test.o
	g++ test.o -o test