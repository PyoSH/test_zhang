# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.24

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

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = "C:\Program Files\JetBrains\CLion 2022.3.1\bin\cmake\win\bin\cmake.exe"

# The command to remove a file.
RM = "C:\Program Files\JetBrains\CLion 2022.3.1\bin\cmake\win\bin\cmake.exe" -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = D:\test_zhang

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = D:\test_zhang\cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/test1.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/test1.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/test1.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/test1.dir/flags.make

CMakeFiles/test1.dir/src/test1.cpp.obj: CMakeFiles/test1.dir/flags.make
CMakeFiles/test1.dir/src/test1.cpp.obj: CMakeFiles/test1.dir/includes_CXX.rsp
CMakeFiles/test1.dir/src/test1.cpp.obj: D:/test_zhang/src/test1.cpp
CMakeFiles/test1.dir/src/test1.cpp.obj: CMakeFiles/test1.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=D:\test_zhang\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/test1.dir/src/test1.cpp.obj"
	C:\mingw32\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/test1.dir/src/test1.cpp.obj -MF CMakeFiles\test1.dir\src\test1.cpp.obj.d -o CMakeFiles\test1.dir\src\test1.cpp.obj -c D:\test_zhang\src\test1.cpp

CMakeFiles/test1.dir/src/test1.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test1.dir/src/test1.cpp.i"
	C:\mingw32\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E D:\test_zhang\src\test1.cpp > CMakeFiles\test1.dir\src\test1.cpp.i

CMakeFiles/test1.dir/src/test1.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test1.dir/src/test1.cpp.s"
	C:\mingw32\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S D:\test_zhang\src\test1.cpp -o CMakeFiles\test1.dir\src\test1.cpp.s

# Object files for target test1
test1_OBJECTS = \
"CMakeFiles/test1.dir/src/test1.cpp.obj"

# External object files for target test1
test1_EXTERNAL_OBJECTS =

test1.exe: CMakeFiles/test1.dir/src/test1.cpp.obj
test1.exe: CMakeFiles/test1.dir/build.make
test1.exe: C:/anaconda3/Library/lib/opencv_gapi401.lib
test1.exe: C:/anaconda3/Library/lib/opencv_stitching401.lib
test1.exe: C:/anaconda3/Library/lib/opencv_aruco401.lib
test1.exe: C:/anaconda3/Library/lib/opencv_bgsegm401.lib
test1.exe: C:/anaconda3/Library/lib/opencv_ccalib401.lib
test1.exe: C:/anaconda3/Library/lib/opencv_dnn_objdetect401.lib
test1.exe: C:/anaconda3/Library/lib/opencv_dpm401.lib
test1.exe: C:/anaconda3/Library/lib/opencv_face401.lib
test1.exe: C:/anaconda3/Library/lib/opencv_fuzzy401.lib
test1.exe: C:/anaconda3/Library/lib/opencv_hfs401.lib
test1.exe: C:/anaconda3/Library/lib/opencv_img_hash401.lib
test1.exe: C:/anaconda3/Library/lib/opencv_line_descriptor401.lib
test1.exe: C:/anaconda3/Library/lib/opencv_reg401.lib
test1.exe: C:/anaconda3/Library/lib/opencv_rgbd401.lib
test1.exe: C:/anaconda3/Library/lib/opencv_saliency401.lib
test1.exe: C:/anaconda3/Library/lib/opencv_stereo401.lib
test1.exe: C:/anaconda3/Library/lib/opencv_structured_light401.lib
test1.exe: C:/anaconda3/Library/lib/opencv_superres401.lib
test1.exe: C:/anaconda3/Library/lib/opencv_surface_matching401.lib
test1.exe: C:/anaconda3/Library/lib/opencv_tracking401.lib
test1.exe: C:/anaconda3/Library/lib/opencv_videostab401.lib
test1.exe: C:/anaconda3/Library/lib/opencv_xfeatures2d401.lib
test1.exe: C:/anaconda3/Library/lib/opencv_xobjdetect401.lib
test1.exe: C:/anaconda3/Library/lib/opencv_xphoto401.lib
test1.exe: C:/anaconda3/Library/lib/opencv_shape401.lib
test1.exe: C:/anaconda3/Library/lib/opencv_phase_unwrapping401.lib
test1.exe: C:/anaconda3/Library/lib/opencv_optflow401.lib
test1.exe: C:/anaconda3/Library/lib/opencv_ximgproc401.lib
test1.exe: C:/anaconda3/Library/lib/opencv_datasets401.lib
test1.exe: C:/anaconda3/Library/lib/opencv_plot401.lib
test1.exe: C:/anaconda3/Library/lib/opencv_text401.lib
test1.exe: C:/anaconda3/Library/lib/opencv_dnn401.lib
test1.exe: C:/anaconda3/Library/lib/opencv_ml401.lib
test1.exe: C:/anaconda3/Library/lib/opencv_video401.lib
test1.exe: C:/anaconda3/Library/lib/opencv_objdetect401.lib
test1.exe: C:/anaconda3/Library/lib/opencv_calib3d401.lib
test1.exe: C:/anaconda3/Library/lib/opencv_features2d401.lib
test1.exe: C:/anaconda3/Library/lib/opencv_flann401.lib
test1.exe: C:/anaconda3/Library/lib/opencv_highgui401.lib
test1.exe: C:/anaconda3/Library/lib/opencv_videoio401.lib
test1.exe: C:/anaconda3/Library/lib/opencv_imgcodecs401.lib
test1.exe: C:/anaconda3/Library/lib/opencv_photo401.lib
test1.exe: C:/anaconda3/Library/lib/opencv_imgproc401.lib
test1.exe: C:/anaconda3/Library/lib/opencv_core401.lib
test1.exe: CMakeFiles/test1.dir/linklibs.rsp
test1.exe: CMakeFiles/test1.dir/objects1.rsp
test1.exe: CMakeFiles/test1.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=D:\test_zhang\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable test1.exe"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\test1.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/test1.dir/build: test1.exe
.PHONY : CMakeFiles/test1.dir/build

CMakeFiles/test1.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\test1.dir\cmake_clean.cmake
.PHONY : CMakeFiles/test1.dir/clean

CMakeFiles/test1.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" D:\test_zhang D:\test_zhang D:\test_zhang\cmake-build-debug D:\test_zhang\cmake-build-debug D:\test_zhang\cmake-build-debug\CMakeFiles\test1.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/test1.dir/depend
