CXX:=g++
#CXXFLAGS:=-std=c++17 -O3 -D BUILDMAIN 
CXXFLAGS:=-std=c++17 -O3 -D BUILDMAIN  -D USE_SIDEEFFECT_API
#CXXFLAGS:=-std=c++17 -g3 -D BUILDMAIN  -D USE_SIDEEFFECT_API

usage:
	@echo make x_sampler
	@echo \   where x.glf is a GLF file.

.SECONDARY:


%_sampler.cpp:%.glf
	./gencpp.py $< -o $@

%_sampler:%_sampler.cpp %_sampler_impl.cpp gramma.hpp
	$(CXX) $(CXXFLAGS) $< -o $@

.PHONY:
clean:
	rm -f *_sampler.cpp *_sampler

.PHONY:
clean_all: clean
	rm -f *_sampler_impl.cpp

