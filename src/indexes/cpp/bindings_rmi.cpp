#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "rmi_cpp.hpp"

namespace py = pybind11;

PYBIND11_MODULE(rmi_cpp, m) {
    m.doc() = "C++ two-stage Recursive Model Index (RMI)";

    py::class_<RecursiveModelIndex>(m, "RecursiveModelIndex")
        .def(py::init<int>(), py::arg("fanout") = 128)
        .def("build", &RecursiveModelIndex::build)
        .def("search", &RecursiveModelIndex::search)
        .def("get_memory_usage", &RecursiveModelIndex::getMemoryUsage);
}
