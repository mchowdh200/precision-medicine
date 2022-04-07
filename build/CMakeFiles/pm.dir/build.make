# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/sdp/precision-medicine

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/sdp/precision-medicine/build

# Include any dependencies generated for this target.
include CMakeFiles/pm.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/pm.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/pm.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/pm.dir/flags.make

CMakeFiles/pm.dir/interface/PMed_OPTANE.cpp.o: CMakeFiles/pm.dir/flags.make
CMakeFiles/pm.dir/interface/PMed_OPTANE.cpp.o: ../interface/PMed_OPTANE.cpp
CMakeFiles/pm.dir/interface/PMed_OPTANE.cpp.o: CMakeFiles/pm.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sdp/precision-medicine/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/pm.dir/interface/PMed_OPTANE.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/pm.dir/interface/PMed_OPTANE.cpp.o -MF CMakeFiles/pm.dir/interface/PMed_OPTANE.cpp.o.d -o CMakeFiles/pm.dir/interface/PMed_OPTANE.cpp.o -c /home/sdp/precision-medicine/interface/PMed_OPTANE.cpp

CMakeFiles/pm.dir/interface/PMed_OPTANE.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/pm.dir/interface/PMed_OPTANE.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sdp/precision-medicine/interface/PMed_OPTANE.cpp > CMakeFiles/pm.dir/interface/PMed_OPTANE.cpp.i

CMakeFiles/pm.dir/interface/PMed_OPTANE.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/pm.dir/interface/PMed_OPTANE.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sdp/precision-medicine/interface/PMed_OPTANE.cpp -o CMakeFiles/pm.dir/interface/PMed_OPTANE.cpp.s

# Object files for target pm
pm_OBJECTS = \
"CMakeFiles/pm.dir/interface/PMed_OPTANE.cpp.o"

# External object files for target pm
pm_EXTERNAL_OBJECTS =

pm: CMakeFiles/pm.dir/interface/PMed_OPTANE.cpp.o
pm: CMakeFiles/pm.dir/build.make
pm: contrib/htslib-install/lib/libhts.a
pm: contrib/zlib-install/lib/libz.a
pm: /usr/lib/gcc/x86_64-linux-gnu/11/libgomp.so
pm: /usr/lib/x86_64-linux-gnu/libpthread.a
pm: CMakeFiles/pm.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/sdp/precision-medicine/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable pm"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/pm.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/pm.dir/build: pm
.PHONY : CMakeFiles/pm.dir/build

CMakeFiles/pm.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/pm.dir/cmake_clean.cmake
.PHONY : CMakeFiles/pm.dir/clean

CMakeFiles/pm.dir/depend:
	cd /home/sdp/precision-medicine/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/sdp/precision-medicine /home/sdp/precision-medicine /home/sdp/precision-medicine/build /home/sdp/precision-medicine/build /home/sdp/precision-medicine/build/CMakeFiles/pm.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/pm.dir/depend

