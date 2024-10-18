# Install script for directory: /mnt/c/Users/marti/OneDrive/Documents/Ecole/EPL/M1/Q1/LELEC2102-Project_Embedded_system/LELEC210X/telecom/hands_on_measurements/gr-fsk/python

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
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/python3/dist-packages/fsk" TYPE FILE FILES
    "/mnt/c/Users/marti/OneDrive/Documents/Ecole/EPL/M1/Q1/LELEC2102-Project_Embedded_system/LELEC210X/telecom/hands_on_measurements/gr-fsk/python/__init__.py"
    "/mnt/c/Users/marti/OneDrive/Documents/Ecole/EPL/M1/Q1/LELEC2102-Project_Embedded_system/LELEC210X/telecom/hands_on_measurements/gr-fsk/python/utils.py"
    "/mnt/c/Users/marti/OneDrive/Documents/Ecole/EPL/M1/Q1/LELEC2102-Project_Embedded_system/LELEC210X/telecom/hands_on_measurements/gr-fsk/python/preamble_detect.py"
    "/mnt/c/Users/marti/OneDrive/Documents/Ecole/EPL/M1/Q1/LELEC2102-Project_Embedded_system/LELEC210X/telecom/hands_on_measurements/gr-fsk/python/flag_detector.py"
    "/mnt/c/Users/marti/OneDrive/Documents/Ecole/EPL/M1/Q1/LELEC2102-Project_Embedded_system/LELEC210X/telecom/hands_on_measurements/gr-fsk/python/synchronization.py"
    "/mnt/c/Users/marti/OneDrive/Documents/Ecole/EPL/M1/Q1/LELEC2102-Project_Embedded_system/LELEC210X/telecom/hands_on_measurements/gr-fsk/python/demodulation.py"
    "/mnt/c/Users/marti/OneDrive/Documents/Ecole/EPL/M1/Q1/LELEC2102-Project_Embedded_system/LELEC210X/telecom/hands_on_measurements/gr-fsk/python/packet_parser.py"
    "/mnt/c/Users/marti/OneDrive/Documents/Ecole/EPL/M1/Q1/LELEC2102-Project_Embedded_system/LELEC210X/telecom/hands_on_measurements/gr-fsk/python/noise_estimation.py"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/python3/dist-packages/fsk" TYPE FILE FILES
    "/mnt/c/Users/marti/OneDrive/Documents/Ecole/EPL/M1/Q1/LELEC2102-Project_Embedded_system/LELEC210X/telecom/hands_on_measurements/gr-fsk/build/python/__init__.pyc"
    "/mnt/c/Users/marti/OneDrive/Documents/Ecole/EPL/M1/Q1/LELEC2102-Project_Embedded_system/LELEC210X/telecom/hands_on_measurements/gr-fsk/build/python/utils.pyc"
    "/mnt/c/Users/marti/OneDrive/Documents/Ecole/EPL/M1/Q1/LELEC2102-Project_Embedded_system/LELEC210X/telecom/hands_on_measurements/gr-fsk/build/python/preamble_detect.pyc"
    "/mnt/c/Users/marti/OneDrive/Documents/Ecole/EPL/M1/Q1/LELEC2102-Project_Embedded_system/LELEC210X/telecom/hands_on_measurements/gr-fsk/build/python/flag_detector.pyc"
    "/mnt/c/Users/marti/OneDrive/Documents/Ecole/EPL/M1/Q1/LELEC2102-Project_Embedded_system/LELEC210X/telecom/hands_on_measurements/gr-fsk/build/python/synchronization.pyc"
    "/mnt/c/Users/marti/OneDrive/Documents/Ecole/EPL/M1/Q1/LELEC2102-Project_Embedded_system/LELEC210X/telecom/hands_on_measurements/gr-fsk/build/python/demodulation.pyc"
    "/mnt/c/Users/marti/OneDrive/Documents/Ecole/EPL/M1/Q1/LELEC2102-Project_Embedded_system/LELEC210X/telecom/hands_on_measurements/gr-fsk/build/python/packet_parser.pyc"
    "/mnt/c/Users/marti/OneDrive/Documents/Ecole/EPL/M1/Q1/LELEC2102-Project_Embedded_system/LELEC210X/telecom/hands_on_measurements/gr-fsk/build/python/noise_estimation.pyc"
    "/mnt/c/Users/marti/OneDrive/Documents/Ecole/EPL/M1/Q1/LELEC2102-Project_Embedded_system/LELEC210X/telecom/hands_on_measurements/gr-fsk/build/python/__init__.pyo"
    "/mnt/c/Users/marti/OneDrive/Documents/Ecole/EPL/M1/Q1/LELEC2102-Project_Embedded_system/LELEC210X/telecom/hands_on_measurements/gr-fsk/build/python/utils.pyo"
    "/mnt/c/Users/marti/OneDrive/Documents/Ecole/EPL/M1/Q1/LELEC2102-Project_Embedded_system/LELEC210X/telecom/hands_on_measurements/gr-fsk/build/python/preamble_detect.pyo"
    "/mnt/c/Users/marti/OneDrive/Documents/Ecole/EPL/M1/Q1/LELEC2102-Project_Embedded_system/LELEC210X/telecom/hands_on_measurements/gr-fsk/build/python/flag_detector.pyo"
    "/mnt/c/Users/marti/OneDrive/Documents/Ecole/EPL/M1/Q1/LELEC2102-Project_Embedded_system/LELEC210X/telecom/hands_on_measurements/gr-fsk/build/python/synchronization.pyo"
    "/mnt/c/Users/marti/OneDrive/Documents/Ecole/EPL/M1/Q1/LELEC2102-Project_Embedded_system/LELEC210X/telecom/hands_on_measurements/gr-fsk/build/python/demodulation.pyo"
    "/mnt/c/Users/marti/OneDrive/Documents/Ecole/EPL/M1/Q1/LELEC2102-Project_Embedded_system/LELEC210X/telecom/hands_on_measurements/gr-fsk/build/python/packet_parser.pyo"
    "/mnt/c/Users/marti/OneDrive/Documents/Ecole/EPL/M1/Q1/LELEC2102-Project_Embedded_system/LELEC210X/telecom/hands_on_measurements/gr-fsk/build/python/noise_estimation.pyo"
    )
endif()

