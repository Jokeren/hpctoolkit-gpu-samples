CC=g++

CXXFLAGS=-fPIC -shared

all: main fib

main: main.cc
	$(CC) -ldl -g -o $@ $<

fib: fib.cc
	$(CC) $(CXXFLAGS) -ldl -g -o $@.so $<

clean:
	rm fib.so main
