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
CMAKE_SOURCE_DIR = /home/sdp/PercisionMedicine

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/sdp/PercisionMedicine/build

# Include any dependencies generated for this target.
include CMakeFiles/PM_pj.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/PM_pj.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/PM_pj.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/PM_pj.dir/flags.make

CMakeFiles/PM_pj.dir/interface/PMed_OPTANE.cpp.o: CMakeFiles/PM_pj.dir/flags.make
CMakeFiles/PM_pj.dir/interface/PMed_OPTANE.cpp.o: ../interface/PMed_OPTANE.cpp
CMakeFiles/PM_pj.dir/interface/PMed_OPTANE.cpp.o: CMakeFiles/PM_pj.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sdp/PercisionMedicine/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/PM_pj.dir/interface/PMed_OPTANE.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/PM_pj.dir/interface/PMed_OPTANE.cpp.o -MF CMakeFiles/PM_pj.dir/interface/PMed_OPTANE.cpp.o.d -o CMakeFiles/PM_pj.dir/interface/PMed_OPTANE.cpp.o -c /home/sdp/PercisionMedicine/interface/PMed_OPTANE.cpp

CMakeFiles/PM_pj.dir/interface/PMed_OPTANE.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/PM_pj.dir/interface/PMed_OPTANE.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sdp/PercisionMedicine/interface/PMed_OPTANE.cpp > CMakeFiles/PM_pj.dir/interface/PMed_OPTANE.cpp.i

CMakeFiles/PM_pj.dir/interface/PMed_OPTANE.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/PM_pj.dir/interface/PMed_OPTANE.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sdp/PercisionMedicine/interface/PMed_OPTANE.cpp -o CMakeFiles/PM_pj.dir/interface/PMed_OPTANE.cpp.s

# Object files for target PM_pj
PM_pj_OBJECTS = \
"CMakeFiles/PM_pj.dir/interface/PMed_OPTANE.cpp.o"

# External object files for target PM_pj
PM_pj_EXTERNAL_OBJECTS =

PM_pj: CMakeFiles/PM_pj.dir/interface/PMed_OPTANE.cpp.o
PM_pj: CMakeFiles/PM_pj.dir/build.make
PM_pj: contrib/htslib-install/lib/libhts.a
PM_pj: contrib/zlib-install/lib/libz.a
PM_pj: /usr/lib/gcc/x86_64-linux-gnu/11/libgomp.so
PM_pj: /usr/lib/x86_64-linux-gnu/libpthread.a
PM_pj: CMakeFiles/PM_pj.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/sdp/PercisionMedicine/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable PM_pj"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/PM_pj.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/PM_pj.dir/build: PM_pj
.PHONY : CMakeFiles/PM_pj.dir/build

CMakeFiles/PM_pj.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/PM_pj.dir/cmake_clean.cmake
.PHONY : CMakeFiles/PM_pj.dir/clean

CMakeFiles/PM_pj.dir/depend:
	cd /home/sdp/PercisionMedicine/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/sdp/PercisionMedicine /home/sdp/PercisionMedicine /home/sdp/PercisionMedicine/build /home/sdp/PercisionMedicine/build /home/sdp/PercisionMedicine/build/CMakeFiles/PM_pj.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/PM_pj.dir/depend

