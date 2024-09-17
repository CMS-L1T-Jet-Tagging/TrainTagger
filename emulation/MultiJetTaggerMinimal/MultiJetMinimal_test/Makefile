.PHONY: clean

MultiJetMinimal_test.so: MultiJetMinimal_emulator_test.o MultiJetMinimal_test.o
	$(CXX) $(CXXFLAGS) $(LD_FLAGS) -shared $^ -o $@

%.o: NN/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm MultiJetMinimal_emulator_test.o MultiJetMinimal_test.o MultiJetMinimal_test.so