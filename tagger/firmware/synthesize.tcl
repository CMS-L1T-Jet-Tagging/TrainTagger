#Setup CMSSW
set CMSSW_BASE "CMSSW_14_0_0_pre3"
# if { [ info exists env(CMSSW_BASE) ] } { set CMSSW_BASE $env(CMSSW_BASE) } { set CMSSW_BASE $env(CMSSW_VERSION) }

#Open the project and then reset it
open_project -reset JetTagger

# Top function
set_top JetTagger
add_files top/JetTagger.cpp -cflags "-std=c++11 -I${CMSSW_BASE}/src"
add_files JetTaggerNN/firmware/JetTaggerNN.cpp -cflags "-std=c++11 -I${CMSSW_BASE}/src"

#Reset solution
open_solution -reset "solution1"

#Specify FPGA and clock constraints
catch {config_array_partition -maximum_size 8192}
config_compile -name_max_length 80
set_part {xcvu13p-flga2577-2-e}
create_clock -period 2.0 -name default

#Synthethize
puts "***** C/RTL SYNTHESIS *****"
set time_start [clock clicks -milliseconds]
csynth_design
set time_end [clock clicks -milliseconds]
puts "C/RTL synthesis took [expr {($time_end - $time_start) / 1000.0}] seconds"
#Make ipbb structure and .dep file later