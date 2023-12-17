# generated with cmake/gen_cmake_skeleton.py, DO NOT MODIFY.
set(PUBLIC_HEADERS
    conformer-activation.h
    conformer-attention.h
    conformer-convolution.h
    conformer-embedding.h
    conformer-encoder.h
    conformer-feedforward.h
    conformer-models.h
    conformer-utils.h
)

add_library(kaldi-conformer SHARED
    conformer-activation.cc
    conformer-attention.cc
    conformer-convolution.cc
    conformer-embedding.cc
    conformer-encoder.cc
    conformer-feedforward.cc
    conformer-models.cc
    conformer-utils.cc
)

target_include_directories(kaldi-conformer PUBLIC 
     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
     $<INSTALL_INTERFACE:include/kaldi>
)

target_link_libraries(kaldi-conformer PUBLIC
    kaldi-matrix
    kaldi-util
    kaldi-types
    matrix-lib
    cu-matrix-lib
    cu-matrix
    cu-math
    kaldi-error
    kaldi-math
    -lcblas
    -llapack
)


install(TARGETS kaldi-conformer
    EXPORT kaldi-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT kaldi
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT kaldi
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT kaldi
)
install(FILES ${PUBLIC_HEADERS} DESTINATION include/kaldi/conformer COMPONENT kaldi)

