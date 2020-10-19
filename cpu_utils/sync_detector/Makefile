CC=g++

CXXFLAGS=-fPIC -shared -lcuda -lcudart

all: sync

sync: sync.cc
	$(CC) $(CXXFLAGS) -ldl -g -o $@.so $<

clean:
	rm *.so
