#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include "transforms.hpp"
#include "dataloader.hpp"
#include "datasets.hpp"

namespace py = boost::python;
namespace np = boost::python::numpy;
BOOST_PYTHON_MODULE(CustomeDataloader)
{
  using namespace boost::python;
  Py_Initialize();
  np::initialize();
  class_<ImagenetDatasets>("ImagenetDatasets",init<std::string,bool>())
    .def("__len__",  &ImagenetDatasets::len);

  
  class_<Transforms>("Transforms",init<bool,bool,bool,int,int,int,float>())
    .def("transform",  &Transforms::transform);
  

  class_<Dataloader>("Dataloader",init<ImagenetDatasets,Transforms,int,int,int,int,int,int,float,bool,bool,bool>())
    .def("next",  &Dataloader::next)
    .def("__len__", &Dataloader::len)
    .def("get_steps_per_epoch",&Dataloader::get_steps_per_epoch);
}
