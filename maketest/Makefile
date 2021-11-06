CC = g++-11
CFLAGS = -Wall -g
INCLUDES = -Iinclude
SRCS = $(wildcard sources/*.cpp)

OBJS := $(subst sources,build,$(SRCS))
OBJS := $(OBJS:.cpp=.o)

MAIN = proggy.out

default: $(MAIN)

$(MAIN): $(OBJS)
	$(CC) $(INCLUDES )$(OBJS) -o $@

build/%.o: sources/%.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

.PHONY: clean

clean:
	rm -f build/*.o $(MAIN)