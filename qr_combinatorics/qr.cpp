#include "qr.h"
#include <functional>

std::vector<std::function<bool(int, int)>> masks = {
    [](int i, int j) -> bool { return j % 3 == 0; },
    [](int i, int j) -> bool { return (i + j) % 3 == 0; },
    [](int i, int j) -> bool { return (i + j) % 2 == 0; },
    [](int i, int j) -> bool { return i % 2 == 0; },
    [](int i, int j) -> bool { return ((i * j) % 3 + i * j) % 2 == 0; },
    [](int i, int j) -> bool { return ((i * j) % 3 + i + j) % 2 == 0; },
    [](int i, int j) -> bool { return (i / 2 + j / 3) % 2 == 0; },
    [](int i, int j) -> bool { return (i * j) % 2 + (i * j) % 3 == 0; }};

uint8_t mask_v(int i, int j, int m) { return masks[m](i, j) ? 1 : 0; }

std::tuple<int, int> index_rotate(int i, int j, int width, int ninety) {
  if (ninety % 4 == 0) {
    return std::make_tuple(i, j);
  } else if (ninety % 4 == 1) {
    return std::make_tuple(j, width - i - 1); // rotate 90
  } else if (ninety % 4 == 2) {
    return std::make_tuple(width - i - 1, width - j - 1); // rotate 180
  }
  return std::make_tuple(width - j - 1, i); // rotate 270
}

extern void cuda_br(std::array<uint8_t, 18>&, int);

void precomp_br(int width, qr_comb_t *comb) {
  for (int i = 0; i < width; i++) {
    for (int j = 0; j < width; j++) {
      printf("%s", comb->extract(j, i) == 1 ? "X" : " ");
    }
    printf("\n");
  }
  std::vector<std::tuple<int, int>> rotate_mask_pairs;
  for (int k = 0; k < 4; k++) {
    for (int m = 0; m < masks.size(); m++) {

      // Get encoding
      printf("(%i, %i): ", k, m);
      uint8_t enc = 0;
      for (int z = 0; z < 4; z++) {
        enc <<= 1;
        auto [i, j] = index_rotate(z % 2, z / 2, width, k + 2);
        uint8_t t =
            (comb->extract(j, i) ^ mask_v(i + width - 1, j + width - 1, m)) & 1;
        printf("%i", t);
        enc += t;
      }
      printf(":%i", enc);
      if (enc == 1) {
        rotate_mask_pairs.push_back(std::make_tuple(k, m));
        printf(" Yes");
      }
      printf("\n");
    }
  }
  for (int i = 0; i < rotate_mask_pairs.size(); i++) {
    int rotate = std::get<0>(rotate_mask_pairs[i]);
    int mask = std::get<1>(rotate_mask_pairs[i]);

    printf("(%i, %i)\n", rotate, mask);
    for (int i = 0; i < width + 2; i++)
      printf("#");
    printf("\n");
    for (int i_ = 0; i_ < width; i_++) {
      printf("#");
      for (int j_ = 0; j_ < width; j_++) {
        auto [i, j] = index_rotate(i_, j_, width, rotate);
        auto [mi, mj] = index_rotate(i_, j_, width, rotate + 2);
        mi += width - 1;
        mj += width - 1;
        uint8_t t =
            (comb->extract(j, i) ^ mask_v(i + width - 1, j + width - 1, mask)) &
            1;
        printf("%s", t == 1 ? "X" : " ");
      }
      printf("#\n");
    }
    for (int i = 0; i < width + 2; i++)
      printf("#");
    printf("\n");

    auto read_codeword_up = [&](int _i, int _j) -> uint8_t{
      uint8_t cw = 0;
      for(int y = 0; y < 4; y++){
        for(int x = 0; x < 2; x++){
          cw <<= 1;
          auto [i, j] = index_rotate(x + _i, y + _j, width, rotate + 2);
          if(i < 0 || j < 0) continue;
          uint8_t t = (comb->extract(i, j) ^ mask_v(j + width - 1, i + width - 1, mask)) & 1;
          cw += t;
          //printf("%s", t == 1 ? "X" : " ");
        }
        //printf("\n");
      }
      //printf("###\n");
      return cw;
    };
    auto read_codeword_down = [&](int _i, int _j) -> uint8_t{
      uint8_t cw = 0;
      for(int y = 3; y >= 0; y--){
        for(int x = 0; x < 2; x++){
          cw <<= 1;
          auto [i, j] = index_rotate(x + _i, y + _j, width, rotate + 2);
          if(i < 0 || j < 0) continue;
          uint8_t t = (comb->extract(i, j) ^ mask_v(j + width - 1, i + width - 1, mask)) & 1;
          cw += t;
          //printf("%s", t == 1 ? "X" : " ");
        }
        //printf("\n");
      }
      //printf("###\n");
      return cw;
    };
    std::array<uint8_t, 18> codewords = {
      read_codeword_up(0, 0),
      read_codeword_up(0, 4),
      read_codeword_up(0, 8), // missing bits 1 and 2
      read_codeword_down(2, 8), // missing bits 128 and 64
      read_codeword_down(2, 4),
      read_codeword_down(2, 0),
      read_codeword_up(4, 0),
      read_codeword_up(4, 4),
      read_codeword_up(4, 8), // missing bits 1 and 2
      read_codeword_down(6, 8), // missing bits 128 and 64
      read_codeword_down(6, 4),
      read_codeword_down(6, 0),
      read_codeword_up(8, 0),
      read_codeword_up(8, 4),
      read_codeword_up(8, 8), // missing bits 1 and 2
      read_codeword_down(10, 8), // missing alot
      read_codeword_down(10, 4), // missing alot
      read_codeword_down(10, 0) // missing alot
    };
    printf("\nCodewords: ");
    for(int i = 0; i < codewords.size(); i++){
      printf("%i ", codewords[i]);
    }
    printf("\n");
    cuda_br(codewords, width);
    break;
  }
}

void qr_comb_t::compute() { precomp_br(width, this); }