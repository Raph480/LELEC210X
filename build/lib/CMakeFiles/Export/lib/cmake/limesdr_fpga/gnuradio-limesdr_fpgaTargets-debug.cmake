#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "gnuradio::gnuradio-limesdr_fpga" for configuration "Debug"
set_property(TARGET gnuradio::gnuradio-limesdr_fpga APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(gnuradio::gnuradio-limesdr_fpga PROPERTIES
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/x86_64-linux-gnu/libgnuradio-limesdr_fpga.so.v1.0-compat-xxx-xunknown"
  IMPORTED_SONAME_DEBUG "libgnuradio-limesdr_fpga.so.1.0.0git"
  )

list(APPEND _IMPORT_CHECK_TARGETS gnuradio::gnuradio-limesdr_fpga )
list(APPEND _IMPORT_CHECK_FILES_FOR_gnuradio::gnuradio-limesdr_fpga "${_IMPORT_PREFIX}/lib/x86_64-linux-gnu/libgnuradio-limesdr_fpga.so.v1.0-compat-xxx-xunknown" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
