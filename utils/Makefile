CC=gcc

CXXFLAGS=-fPIC -shared

hook.so: hook.c
	$(CC) $(CXXFLAGS) -ldl -o $@ $<

clean:
	rm hook.so
