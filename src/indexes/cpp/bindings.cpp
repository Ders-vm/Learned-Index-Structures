#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "btree_cpp.hpp"

namespace py = pybind11;

PYBIND11_MODULE(btree_cpp, m) {
    m.doc() = "C++ B-Tree baseline implementation (for comparison with learned indexes)";

    py::class_<BTree>(m, "BTree")
        .def(py::init<int>(), py::arg("order") = 128)
        // ✅ Expose both method names
        .def("build", &BTree::build, py::arg("keys"),
             "Build the B-Tree from a sorted list of keys.")
        .def("build_from_sorted_array", &BTree::build, py::arg("keys"),
             "Alias for compatibility with Python version.")
        .def("search", &BTree::search, py::arg("key"),
             "Search for a key; returns (found, comparisons).")
        .def("get_memory_usage", &BTree::getMemoryUsage,
             "Approximate total memory usage in bytes.");
}
