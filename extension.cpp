#include "transforms.hpp"
#include "datasets.hpp"
#include "dataloader.hpp"
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(customdl, m) {
    Py_Initialize();
    py::class_<ImagenetDatasets>(m, "ImagenetDatasets")
      .def(py::init<const std::string&, bool>())
      .def("__len__", &ImagenetDatasets::len);
  
    py::class_<Transforms>(m, "Transforms")
      .def(py::init<bool,bool,bool,int,int,int,float>());

    
    py::class_<Dataloader>(m, "Dataloader")
      .def(py::init<ImagenetDatasets,Transforms,int,int,int,float,bool,bool,bool>())
      .def("__len__", &Dataloader::len)
      .def("next",  &Dataloader::next);

    py::class_<Batch>(m, "Batch")
      .def("image", &Batch::image)
      .def("label", &Batch::label);
  

}