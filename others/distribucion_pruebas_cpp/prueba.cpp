#include <bits/stdc++.h>
#include <omp.h>

int main(){
  std::vector<std::vector<std::set<int>>> v(100, std::vector<std::set<int>>(16, std::set<int>()));

  for(size_t i = 0; i < v.size(); ++i){
    #pragma omp parallel for num_threads(16)
    for(size_t index = 0; index < 10000; ++index){
      v[i][omp_get_num_threads()].insert(index);
    }
  }

  // for(size_t i = 0; i < v.size() - 1; ++i){
  //   for(size_t index = 0; index < v[i].size(); ++index){
  //     if(v[i][index] != v[i+1][index]){
  //       std::cout << "different\n";
  //     }
  //   }
  // }

  return 0;
}