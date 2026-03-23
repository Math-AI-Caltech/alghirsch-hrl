#include <ATen/Operators.h>
#include <Python.h>
#include <omp.h>
#include <torch/all.h>
#include <torch/library.h>

extern "C" {
PyObject *PyInit__C(void) {
  static struct PyModuleDef module_def = {PyModuleDef_HEAD_INIT, "_C", NULL, -1,
                                          NULL};
  return PyModule_Create(&module_def);
}
}

at::Tensor shortest_path_cpu(const at::Tensor &x,
                             const at::Tensor &edge_index) {
  // TODO Check that the shapes of x and edge_index
  TORCH_CHECK(x.dtype() == at::kLong);
  TORCH_CHECK(edge_index.dtype() == at::kLong);
  TORCH_INTERNAL_ASSERT(x.device().type() == at::DeviceType::CPU);
  TORCH_INTERNAL_ASSERT(edge_index.device().type() == at::DeviceType::CPU);

  int64_t num_vertices = x.size(0);

  at::Tensor x_contg = x.contiguous();
  at::Tensor edge_index_contg = edge_index.contiguous();
  int64_t *edge_index_ptr = edge_index_contg.data_ptr<int64_t>();
  at::Tensor dist_mat = (torch::zeros({num_vertices, num_vertices}) +
                         std::numeric_limits<float>::infinity())
                            .contiguous();
  float *dist_mat_ptr = dist_mat.data_ptr<float>();

  for (int64_t i = 0; i < edge_index_contg.size(1); ++i) {
    int64_t s = edge_index_ptr[i];
    int64_t t = edge_index_ptr[i + edge_index_contg.size(1)];

    dist_mat_ptr[s + t * num_vertices] = 1.;
    dist_mat_ptr[t + s * num_vertices] = 1.;
  }

  for (int64_t i = 0; i < num_vertices; ++i) dist_mat_ptr[i + i * num_vertices] = 0.;

  for (int64_t k = 0; k < num_vertices; ++k) {
    for (int64_t i = 0; i < num_vertices; ++i) {
      for (int64_t j = 0; j < num_vertices; ++j) {
        dist_mat_ptr[i + j * num_vertices] =
            fmin(dist_mat_ptr[i + j * num_vertices],
                 dist_mat_ptr[i + k * num_vertices] +
                     dist_mat_ptr[k + j * num_vertices]);
      }
    }
  }

  return dist_mat;
}

at::Tensor shortest_path_cpu_parallel(const at::Tensor &x,
                                      const at::Tensor &edge_index) {
  // Make contiguous
  at::Tensor x_contg = x.contiguous();
  at::Tensor edge_index_contg = edge_index.contiguous();

  // Get batch dimension
  int64_t num_batches = x_contg.size(0);
  int64_t num_vertices = x_contg.size(1);
  auto options =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

  // Reserve memory for matrices
  at::Tensor distances =
      torch::full({num_batches, num_vertices, num_vertices},
                  std::numeric_limits<float>::infinity(), options);

// Run algorithm in parallel
#pragma omp parallel for
  for (int i = 0; i < num_batches; ++i) {
    distances[i] = shortest_path_cpu(x_contg[i], edge_index_contg[i]);
  }

  return distances;
}

TORCH_LIBRARY(diameter, m) {
  m.def("shortest_path(Tensor x, Tensor edge_index) -> Tensor");
  m.def("shortest_path_parallel(Tensor x, Tensor edge_index) -> Tensor");
}

TORCH_LIBRARY_IMPL(diameter, CPU, m) {
  m.impl("shortest_path", &shortest_path_cpu);
  m.impl("shortest_path_parallel", &shortest_path_cpu_parallel);
}
