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

// extern void cuda_br(std::array<uint8_t, 18>&, int);

void cpu_br(std::array<uint8_t, 18> &data, int rotate, std::vector<uint16_t>&);

void precomp_br(int width, qr_comb_t *comb, std::vector<uint16_t>& values) {
  std::vector<std::tuple<int, int>> rotate_mask_pairs;
  for (int k = 0; k < 4; k++) {
    for (int m = 0; m < masks.size(); m++) {

      // Get encoding
      uint8_t enc = 0;
      for (int z = 0; z < 4; z++) {
        enc <<= 1;
        auto [i, j] = index_rotate(z % 2, z / 2, width, k + 2);
        uint8_t t =
            (comb->extract(j, i) ^ mask_v(i + width - 1, j + width - 1, m)) & 1;
        enc += t;
      }
      if (enc == 1) {
        rotate_mask_pairs.push_back(std::make_tuple(k, m));
      }
    }
  }
  for (int i = 0; i < rotate_mask_pairs.size(); i++) {
    int rotate = std::get<0>(rotate_mask_pairs[i]);
    int mask = std::get<1>(rotate_mask_pairs[i]);

    auto masked_bit = [&](int i_, int j_) {
      auto [i, j] = index_rotate(i_, j_, width, rotate + 2);
      return (comb->extract(j, i) ^
              mask_v(i_ + width - 1, j_ + width - 1, mask)) &
             1;
    };
    /*
    printf("-----------");
    for(int i = 0; i < 11; i++){
      for(int j = 0; j < 11; j++){
        auto t = (*comb)(j, i);
        printf("%s", t ? "#" : " ");
      }
      printf("\n");
    }
    printf("-----------");
    */
    auto read_codeword_up = [&](int _i, int _j) -> uint8_t {
      if (_i == 0 && _j == 0)
        ;
      uint8_t cw = 0;
      for (int y = 0; y < 4; y++) {
        for (int x = 0; x < 2; x++) {
          cw <<= 1;
          auto [i, j] = index_rotate(x + _i, y + _j, width, rotate + 2);
          if (i < 0 || j < 0)
            continue;
          auto t = masked_bit(y + _j, x + _i);
          cw += t;
        }
      }
      return cw;
    };
    auto read_codeword_down = [&](int _i, int _j) -> uint8_t {
      uint8_t cw = 0;
      for (int y = 3; y >= 0; y--) {
        for (int x = 0; x < 2; x++) {
          cw <<= 1;
          auto t = masked_bit(y + _j, x + _i);
          cw += t;
        }
      }
      return cw;
    };
    std::array<uint8_t, 18> codewords = {
        read_codeword_up(0, 0),    read_codeword_up(0, 4),
        read_codeword_up(0, 8),   // missing bits 1 and 2
        read_codeword_down(2, 8), // missing bits 128 and 64
        read_codeword_down(2, 4),  read_codeword_down(2, 0),
        read_codeword_up(4, 0),    read_codeword_up(4, 4),
        read_codeword_up(4, 8),   // missing bits 1 and 2
        read_codeword_down(6, 8), // missing bits 128 and 64
        read_codeword_down(6, 4),  read_codeword_down(6, 0),
        read_codeword_up(8, 0),    read_codeword_up(8, 4),
        read_codeword_up(8, 8),    // missing bits 1 and 2
        read_codeword_down(10, 8), // missing alot
        read_codeword_down(10, 4), // missing alot
        read_codeword_down(10, 0)  // missing alot
    };
    cpu_br(codewords, width, values);
  }
}

template <int i>
int extract(const std::array<uint8_t, i> &bytes, int pos, int len) {
  int shift = 24 - (pos & 7) - len;
  int mask = (1 << len) - 1;
  int byteIndex = pos >> 3;
  return (((bytes[byteIndex] << 16) | (bytes[++byteIndex] << 8) |
           bytes[++byteIndex]) >>
          shift) &
         mask;
}

std::pair<bool, uint16_t> extract_numeric(const std::array<uint8_t, 14> &bytes,
                                          int bit_idx) {
  int n = extract<14>(bytes, bit_idx, 10);
  if (n != 4)
    return std::make_pair(false, n);
  bit_idx += 10;
  int x = extract<14>(bytes, bit_idx, 10);
  bit_idx += 10;
  uint16_t k = x;
  if(k >= 1000) 
    return std::make_pair(false, k);
  x = extract<14>(bytes, bit_idx, 4);
  bit_idx += 4;
  if (x >= 10)
    return std::make_pair(false, x);
  uint8_t d = x;
  return std::make_pair(true, k * 10 + d);
}

void compute_num(const std::array<uint8_t, 18> &data, uint8_t changes,
                 std::vector<uint16_t> &answers) {
  std::array<uint8_t, 14> working;
  for (int i = 0; i < 14; i++) {
    working[i] = data[i];
  }

  working[2] |= ((changes) & (1)) << 1;
  working[2] |= ((changes >> (1)) & 1);
  working[3] |= ((changes >> (2)) & 1) << 7;
  working[3] |= ((changes >> (3)) & 1) << 6;

  int bit_idx = 4;
  int mode = extract<14>(working, 0, 4);
  if (mode != 1)
    return;

  auto [yeah, x] = extract_numeric(working, bit_idx);
  if (yeah) {
    for (int i = 0; i < answers.size(); i++) {
      if (answers[i] == x)
        return;
    }
    answers.push_back(x); // TODO yeah, its not working
  }
}

void cpu_br(std::array<uint8_t, 18> &data, int width, std::vector<uint16_t>& values) {
  /*
  printf("Keywords: ");
  for(int i = 0; i < 18; i++){
    printf("%s%i", i == 0 ? "" : ", ", data[i]);
  }
  printf("\n");
  */
  for (uint8_t i = 0; i <= 0x0f; i++) {
    compute_num(data, i, values);
  }
}

void qr_comb_t::compute(std::vector<uint16_t>& values) { precomp_br(width, this, values); }
