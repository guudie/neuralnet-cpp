.PHONY: all
all: test
# This rule tells make how to build hello from test1.cpp
test: test.cpp
	g++ -O2 -I../src test.cpp -o test.o

# This rule tells make to delete the program
.PHONY: clean 
clean:
	rm -f test.o