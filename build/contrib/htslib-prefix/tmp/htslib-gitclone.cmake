
if(NOT "/home/sdp/precision-medicine/build/contrib/htslib-prefix/src/htslib-stamp/htslib-gitinfo.txt" IS_NEWER_THAN "/home/sdp/precision-medicine/build/contrib/htslib-prefix/src/htslib-stamp/htslib-gitclone-lastrun.txt")
  message(STATUS "Avoiding repeated git clone, stamp file is up to date: '/home/sdp/precision-medicine/build/contrib/htslib-prefix/src/htslib-stamp/htslib-gitclone-lastrun.txt'")
  return()
endif()

execute_process(
  COMMAND ${CMAKE_COMMAND} -E rm -rf "/home/sdp/precision-medicine/build/contrib/htslib-prefix/src/htslib"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to remove directory: '/home/sdp/precision-medicine/build/contrib/htslib-prefix/src/htslib'")
endif()

# try the clone 3 times in case there is an odd git clone issue
set(error_code 1)
set(number_of_tries 0)
while(error_code AND number_of_tries LESS 3)
  execute_process(
    COMMAND "/usr/bin/git"  clone --no-checkout --config "advice.detachedHead=false" "https://github.com/nsubtil/htslib.git" "htslib"
    WORKING_DIRECTORY "/home/sdp/precision-medicine/build/contrib/htslib-prefix/src"
    RESULT_VARIABLE error_code
    )
  math(EXPR number_of_tries "${number_of_tries} + 1")
endwhile()
if(number_of_tries GREATER 1)
  message(STATUS "Had to git clone more than once:
          ${number_of_tries} times.")
endif()
if(error_code)
  message(FATAL_ERROR "Failed to clone repository: 'https://github.com/nsubtil/htslib.git'")
endif()

execute_process(
  COMMAND "/usr/bin/git"  checkout fix-signed-one-bit-bitfield --
  WORKING_DIRECTORY "/home/sdp/precision-medicine/build/contrib/htslib-prefix/src/htslib"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to checkout tag: 'fix-signed-one-bit-bitfield'")
endif()

set(init_submodules TRUE)
if(init_submodules)
  execute_process(
    COMMAND "/usr/bin/git"  submodule update --recursive --init 
    WORKING_DIRECTORY "/home/sdp/precision-medicine/build/contrib/htslib-prefix/src/htslib"
    RESULT_VARIABLE error_code
    )
endif()
if(error_code)
  message(FATAL_ERROR "Failed to update submodules in: '/home/sdp/precision-medicine/build/contrib/htslib-prefix/src/htslib'")
endif()

# Complete success, update the script-last-run stamp file:
#
execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy
    "/home/sdp/precision-medicine/build/contrib/htslib-prefix/src/htslib-stamp/htslib-gitinfo.txt"
    "/home/sdp/precision-medicine/build/contrib/htslib-prefix/src/htslib-stamp/htslib-gitclone-lastrun.txt"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to copy script-last-run stamp file: '/home/sdp/precision-medicine/build/contrib/htslib-prefix/src/htslib-stamp/htslib-gitclone-lastrun.txt'")
endif()

