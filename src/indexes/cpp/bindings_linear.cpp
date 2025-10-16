#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "linear_model_cpp.hpp"

namespace py = pybind11;

PYBIND11_MODULE(linear_model_cpp, m) {
    m.doc() = "C++ Linear Model Index (Learned Index baseline)";

    py::class_<LinearModelIndex>(m, "LinearModelIndex")
        .def(py::init<>())
        .def("build", &LinearModelIndex::build)
        .def("search", &LinearModelIndex::search)
        .def("get_memory_usage", &LinearModelIndex::getMemoryUsage);
}
