CXX      ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall -Wextra -Werror -fPIC

all: libsamodel.so selftest perf_model

libsamodel.so: sa_model.cpp sa_model.hpp
	$(CXX) $(CXXFLAGS) -shared -o $@ sa_model.cpp

selftest: selftest.cpp sa_model.cpp sa_model.hpp
	$(CXX) $(CXXFLAGS) -o $@ selftest.cpp sa_model.cpp

perf_model: perf_model.cpp sa_model.cpp sa_model.hpp
	$(CXX) $(CXXFLAGS) -o $@ perf_model.cpp sa_model.cpp

test: selftest
	./selftest

clean:
	rm -f libsamodel.so selftest perf_model

.PHONY: all test clean
