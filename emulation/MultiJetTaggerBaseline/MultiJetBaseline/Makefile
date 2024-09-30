.PHONY: clean

MultiJetBaseline.so: MultiJetMinimal_emulator_test.o MultiJetBaseline.o
	$(CXX) $(CXXFLAGS) $(LD_FLAGS) -shared $^ -o $@

%.o: NN/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm MultiJetMinimal_emulator_test.o MultiJetBaseline.o MultiJetBaseline.so