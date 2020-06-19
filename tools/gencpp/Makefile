CXX:=g++
CXXFLAGS:=-std=c++17 -O3

usage:
	@echo make x_sampler
	@echo \   where x.glf is a GLF file.

.SECONDARY:


%_sampler.cpp:%.glf
	./gencpp.py $< -o $@

%_sampler:%_sampler.cpp gramma.hpp
	$(CXX) $(CXXFLAGS) -D BUILDMAIN $< -o $@

clean:
	rm -f *_sampler.cpp *_sampler
