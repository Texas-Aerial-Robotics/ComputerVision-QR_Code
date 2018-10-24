#include "qr.h"
#include <functional>

std::vector<std::function<bool(int, int)> > masks = {
  [](int i, int j) -> bool {return j % 3 == 0;},
  [](int i, int j) -> bool {return (i + j) % 3 == 0;},
  [](int i, int j) -> bool {return (i + j) % 2 == 0;},
  [](int i, int j) -> bool {return i % 2 == 0;},
  [](int i, int j) -> bool {return ((i * j) % 3 + i * j) % 2 == 0;},
  [](int i, int j) -> bool {return ((i * j) % 3 + i + j) % 2 == 0;},
  [](int i, int j) -> bool {return (i / 2 + j / 3) % 2 == 0;},
  [](int i, int j) -> bool {return (i * j) % 2 + (i * j) % 3 == 0;}
};

uint8_t mask_v(int i, int j, int m){
  return masks[m](i, j) ? 1 : 0;
}

std::tuple<int, int> index_rotate(int i, int j, int width, int ninety){
  if(ninety % 4 == 0) {
    return std::make_tuple(i, j);
  } else if(ninety % 4 == 1) {
    return std::make_tuple(j, width - i - 1); // rotate 90
  } else if(ninety % 4 == 2) {
    return std::make_tuple(width - i - 1, width - j - 1); // rotate 180
  }
  return std::make_tuple(width - j - 1, i); // rotate 270
}

void precomp_br(int width, qr_comb_t *comb){
  for(int i = 0; i < width; i++){
    for(int j = 0; j < width; j++){
      printf("%s", comb->extract(j, i) == 1 ? "X" : " ");
    }
    printf("\n");
  }
  std::vector<std::tuple<int, int> > rotate_mask_pairs;
  for(int k = 0; k < 4; k++){
    for(int m = 0; m < masks.size(); m++){

      // Get encoding
      printf("(%i, %i): ", k, m);
      uint8_t enc = 0;
      for(int z = 0; z < 4; z++){
        enc <<= 1;
        auto [i, j] = index_rotate(z % 2, z / 2, width, k + 2);
        uint8_t t = (comb->extract(j, i) ^ mask_v(i + width - 1, j + width - 1, m)) & 1;
        printf("%i", t);
        enc += t;
      }
      printf(":%i", enc);
      if(enc == 1) {
        rotate_mask_pairs.push_back(std::make_tuple(k, m));
        printf(" Yes");
      }
      printf("\n");
    }
  }
  for(int i = 0; i < rotate_mask_pairs.size(); i++){
    int rotate = std::get<0>(rotate_mask_pairs[i]);
    int mask = std::get<1>(rotate_mask_pairs[i]);
    printf("(%i, %i)\n", rotate, mask);
    for(int i = 0; i < width + 2; i++) printf("#");
    printf("\n");
    for(int i_ = 0; i_ < width; i_++){
      printf("#");
      for(int j_ = 0; j_ < width; j_++){
        auto [i, j] = index_rotate(i_, j_, width, rotate);
        auto [mi, mj] = index_rotate(i_, j_, width, rotate + 2);
        mi += width - 1;
        mj += width - 1;
        uint8_t t = (comb->extract(j, i) ^ mask_v(i + width - 1, j + width - 1, mask)) & 1;
        printf("%s", t == 1 ? "X" : " ");
      }
      printf("#\n");
    }
    for(int i = 0; i < width + 2; i++) printf("#");
    printf("\n");
  }
}

void qr_comb_t::compute(){
  precomp_br(width, this);
}