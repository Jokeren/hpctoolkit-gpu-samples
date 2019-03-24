CC=gcc

CXXFLAGS=-fPIC -shared

all: env dl-tls pthread test-tls-1 test-tls-2

env: env.c
	$(CC) $(CXXFLAGS) -ldl -g -o $@.so $<

dl-tls: dl-tls.c
	$(CC) $(CXXFLAGS) -ldl -g -o $@.so $<

pthread: pthread.c
	$(CC) $(CXXFLAGS) -ldl -g -o $@.so $<

test-tls-1: test-tls-1.c
	$(CC) $(CXXFLAGS) -ldl -g -o $@.so $<

test-tls-2: test-tls-2.c
	$(CC) $(CXXFLAGS) -ldl -g -o $@.so $<

clean:
	rm env.so dl-tls.so pthread.so test-tls-1.so test-tls-2.so
