.PHONY: clean

MultiJetBaseline.so: MultiJetBaseline_emulator.o MultiJetBaseline.o
	$(CXX) $(CXXFLAGS) $(LD_FLAGS) -shared $^ -o $@

%.o: NN/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm MultiJetBaseline_emulator.o MultiJetBaseline.o MultiJetBaseline.so
