# Install script for directory: /mnt/c/Users/marti/OneDrive/Documents/Ecole/EPL/M1/Q1/LELEC2102-Project_Embedded_system/LELEC210X/telecom/hands_on_measurements/gr-fsk/grc

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/gnuradio/grc/blocks" TYPE FILE FILES
    "/mnt/c/Users/marti/OneDrive/Documents/Ecole/EPL/M1/Q1/LELEC2102-Project_Embedded_system/LELEC210X/telecom/hands_on_measurements/gr-fsk/grc/fsk_preamble_detect.block.yml"
    "/mnt/c/Users/marti/OneDrive/Documents/Ecole/EPL/M1/Q1/LELEC2102-Project_Embedded_system/LELEC210X/telecom/hands_on_measurements/gr-fsk/grc/fsk_flag_detector.block.yml"
    "/mnt/c/Users/marti/OneDrive/Documents/Ecole/EPL/M1/Q1/LELEC2102-Project_Embedded_system/LELEC210X/telecom/hands_on_measurements/gr-fsk/grc/fsk_synchronization.block.yml"
    "/mnt/c/Users/marti/OneDrive/Documents/Ecole/EPL/M1/Q1/LELEC2102-Project_Embedded_system/LELEC210X/telecom/hands_on_measurements/gr-fsk/grc/fsk_demodulation.block.yml"
    "/mnt/c/Users/marti/OneDrive/Documents/Ecole/EPL/M1/Q1/LELEC2102-Project_Embedded_system/LELEC210X/telecom/hands_on_measurements/gr-fsk/grc/fsk_packet_parser.block.yml"
    "/mnt/c/Users/marti/OneDrive/Documents/Ecole/EPL/M1/Q1/LELEC2102-Project_Embedded_system/LELEC210X/telecom/hands_on_measurements/gr-fsk/grc/fsk_noise_estimation.block.yml"
    )
endif()
