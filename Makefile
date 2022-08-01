SOURCES := $(wildcard *.c *.cpp *.cc)
OBJECTS := $(addsuffix .o,$(basename $(SOURCES)))

LDLIBS +=

# if any -O... is requested, use that; otherwise, do -O3
FLAGS_OPTIMIZATION := $(if $(filter -O%,$(CFLAGS) $(CXXFLAGS) $(CPPFLAGS)),,-O3)
CPPFLAGS := -MMD -g $(FLAGS_OPTIMIZATION) -Wall -Wextra -Wno-missing-field-initializers
CFLAGS += -std=gnu11 -fno-omit-frame-pointer
CXXFLAGS += -std=gnu++11

test: $(OBJECTS)
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

clean:
	rm -rf target *.o *.d
.PHONY: clean

-include *.d
