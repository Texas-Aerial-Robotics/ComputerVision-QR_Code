    cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
    project(cmake_and_cuda LANGUAGES CXX CUDA)
     
    include(CTest)
    
    execute_process (
      COMMAND bash -c "cd build;pwd -P"
      OUTPUT_VARIABLE AWD
    )
     
    add_library(qr STATIC ${AWD}/../qr_combinatorics.cu)
     
    # Request that qr be built with -std=c++11
    # As this is a public compile feature anything that links to 
    # qr will also build with -std=c++11
    target_compile_features(qr PUBLIC cxx_std_11)
    
    string(APPEND CMAKE_CUDA_FLAGS " -arch=sm_30")
     
    # We need to explicitly state that we need all CUDA files in the 
    # qr library to be built with -dc as the member functions 
    # could be called by other libraries and executables
    set_target_properties( qr
                           PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
     
    add_executable(qr_comb execute.cpp qr.cpp)
    
    set_target_properties(qr_comb PROPERTIES
            CXX_STANDARD 17
	    )

    set_property(TARGET qr_comb 
                 PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(qr_comb PRIVATE qr)
     
    if(APPLE)
      # We need to add the path to the driver (libcuda.dylib) as an rpath, 
      # so that the static cuda runtime can find it at runtime.
      set_property(TARGET qr_comb 
                   PROPERTY
                   BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    endif()
