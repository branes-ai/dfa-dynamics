# domain flow graph tests
add_subdirectory(rdg)
add_subdirectory(dfg)

if(DOMAINFLOW_VISUALIZATION)
add_subdirectory(viz)
endif()

# importers
#add_subdirectory(opt)
#add_subdirectory(import)
