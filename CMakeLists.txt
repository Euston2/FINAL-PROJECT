cmake_minimum_required(VERSION 3.15)
project(EDEMS_Capture CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find PcapPlusPlus (path supplied via -DPcapPlusPlus_ROOT=... at configure time)
find_package(PcapPlusPlus CONFIG REQUIRED)

# Create our executable
add_executable(edems_capture main.cpp)

# Link the PcapPlusPlus library to our executable.
# PcapPlusPlus is actually three libraries (Common++, Packet++, Pcap++),
# not one umbrella target -- link all three.
target_link_libraries(edems_capture PRIVATE
    PcapPlusPlus::Pcap++
    PcapPlusPlus::Packet++
    PcapPlusPlus::Common++
)