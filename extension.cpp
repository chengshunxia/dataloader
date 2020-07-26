
#include "transforms.hpp"
#include "dataloader.hpp"
#include "datasets.hpp"
#include <boost/python.hpp>


BOOST_PYTHON_MODULE(CustomeDataloader)
{
  using namespace boost::python;
  class_<ImagenetDatasets>("ImagenetDatasets",init<std::string,bool>())
    .def("__len__",  &ImagenetDatasets::len);

  
  class_<Transforms>("Transforms",init<bool,bool,bool,int,int,int,float>())
    .def("transform",  &Transforms::transform);
  

  class_<Dataloader>("Dataloader",init<ImagenetDatasets,Transforms,int,int,int,int,int,int,float,bool,bool,bool>())
    .def("next",  &Dataloader::next)
    .def("__len__", &Dataloader::len);
  
}