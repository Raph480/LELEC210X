# Install script for directory: /home/martin/Documents/EPL/M1/Project-Embedded/LELEC210X/telecom/hands_on_measurements/gr-fsk/python

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

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/python3.10/dist-packages/fsk" TYPE FILE FILES
    "/home/martin/Documents/EPL/M1/Project-Embedded/LELEC210X/telecom/hands_on_measurements/gr-fsk/python/__init__.py"
    "/home/martin/Documents/EPL/M1/Project-Embedded/LELEC210X/telecom/hands_on_measurements/gr-fsk/python/utils.py"
    "/home/martin/Documents/EPL/M1/Project-Embedded/LELEC210X/telecom/hands_on_measurements/gr-fsk/python/preamble_detect.py"
    "/home/martin/Documents/EPL/M1/Project-Embedded/LELEC210X/telecom/hands_on_measurements/gr-fsk/python/flag_detector.py"
    "/home/martin/Documents/EPL/M1/Project-Embedded/LELEC210X/telecom/hands_on_measurements/gr-fsk/python/synchronization.py"
    "/home/martin/Documents/EPL/M1/Project-Embedded/LELEC210X/telecom/hands_on_measurements/gr-fsk/python/demodulation.py"
    "/home/martin/Documents/EPL/M1/Project-Embedded/LELEC210X/telecom/hands_on_measurements/gr-fsk/python/packet_parser.py"
    "/home/martin/Documents/EPL/M1/Project-Embedded/LELEC210X/telecom/hands_on_measurements/gr-fsk/python/onQuery_noise_estimation.py"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/python3.10/dist-packages/fsk" TYPE FILE FILES
    "/home/martin/Documents/EPL/M1/Project-Embedded/LELEC210X/telecom/hands_on_measurements/gr-fsk/build/python/__init__.pyc"
    "/home/martin/Documents/EPL/M1/Project-Embedded/LELEC210X/telecom/hands_on_measurements/gr-fsk/build/python/utils.pyc"
    "/home/martin/Documents/EPL/M1/Project-Embedded/LELEC210X/telecom/hands_on_measurements/gr-fsk/build/python/preamble_detect.pyc"
    "/home/martin/Documents/EPL/M1/Project-Embedded/LELEC210X/telecom/hands_on_measurements/gr-fsk/build/python/flag_detector.pyc"
    "/home/martin/Documents/EPL/M1/Project-Embedded/LELEC210X/telecom/hands_on_measurements/gr-fsk/build/python/synchronization.pyc"
    "/home/martin/Documents/EPL/M1/Project-Embedded/LELEC210X/telecom/hands_on_measurements/gr-fsk/build/python/demodulation.pyc"
    "/home/martin/Documents/EPL/M1/Project-Embedded/LELEC210X/telecom/hands_on_measurements/gr-fsk/build/python/packet_parser.pyc"
    "/home/martin/Documents/EPL/M1/Project-Embedded/LELEC210X/telecom/hands_on_measurements/gr-fsk/build/python/onQuery_noise_estimation.pyc"
    "/home/martin/Documents/EPL/M1/Project-Embedded/LELEC210X/telecom/hands_on_measurements/gr-fsk/build/python/__init__.pyo"
    "/home/martin/Documents/EPL/M1/Project-Embedded/LELEC210X/telecom/hands_on_measurements/gr-fsk/build/python/utils.pyo"
    "/home/martin/Documents/EPL/M1/Project-Embedded/LELEC210X/telecom/hands_on_measurements/gr-fsk/build/python/preamble_detect.pyo"
    "/home/martin/Documents/EPL/M1/Project-Embedded/LELEC210X/telecom/hands_on_measurements/gr-fsk/build/python/flag_detector.pyo"
    "/home/martin/Documents/EPL/M1/Project-Embedded/LELEC210X/telecom/hands_on_measurements/gr-fsk/build/python/synchronization.pyo"
    "/home/martin/Documents/EPL/M1/Project-Embedded/LELEC210X/telecom/hands_on_measurements/gr-fsk/build/python/demodulation.pyo"
    "/home/martin/Documents/EPL/M1/Project-Embedded/LELEC210X/telecom/hands_on_measurements/gr-fsk/build/python/packet_parser.pyo"
    "/home/martin/Documents/EPL/M1/Project-Embedded/LELEC210X/telecom/hands_on_measurements/gr-fsk/build/python/onQuery_noise_estimation.pyo"
    )
endif()

