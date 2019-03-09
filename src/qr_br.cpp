#include <functional>

#include "qr.h"

std::vector<std::function<bool(int, int)>> masks = {
    [](int i, int j) -> bool { return j % 3 == 0; },
    [](int i, int j) -> bool { return (i + j) % 3 == 0; },
    [](int i, int j) -> bool { return (i + j) % 2 == 0; },
    [](int i, int j) -> bool { return i % 2 == 0; },
    [](int i, int j) -> bool { return ((i * j) % 3 + i * j) % 2 == 0; },
    [](int i, int j) -> bool { return ((i * j) % 3 + i + j) % 2 == 0; },
    [](int i, int j) -> bool { return (i / 2 + j / 3) % 2 == 0; },
    [](int i, int j) -> bool { return (i * j) % 2 + (i * j) % 3 == 0; }};

void comp(int width, qr::qr_t q, std::vector<uint16_t>& dat){
  for(int orientation = 0; orientation < 4; orientation++){
    auto code_o = [&](int i, int j) {
      switch (orientation){
      case 0:
        return q(i, j);
      case 1:
        return q(j, width - i - 1);
      case 2:
        return q(width - i - 1, width - j - 1);
      default:
        return q(width - j - 1, i);
      }
    };
    for(int mask = 0; mask < masks.size(); mask++){
      auto code_m = [&](int i, int j){
        return masks[mask](i + width - 1, j + width - 1) ^ code_o(i, j);
      };
      auto read_codeword_up = [&](int i, int j){
        uint8_t codeword = 0;
        for(int k = 0; k < 4; k++){
          for(int z = 0; z < 2; z++){
            codeword <<= 1;
            if(i - k < 0 || j - z < 0) continue;
            codeword += (code_m(i - k, j - z) ? 1 : 0);
          }
        }
        return codeword;
      };
      auto read_codeword_down = [&](int i, int j){
        uint8_t codeword = 0;
        for(int k = 3; k >= 0; k--){
          for(int z = 0; z < 2; z++){
            codeword <<= 1;
            if(i - k < 0 || j - z < 0) continue;
            codeword += (code_m(i - k, j - z) ? 1 : 0);
          }
        }
        return codeword;
      };
      std::vector<uint8_t> codewords;
      for(int c = width - 1; c > 0; c -= 2){
        for(int r = width - 1; r > 0; r -= 4){
          if((c % 4) / 2 == ((width - 1) % 4) / 2){
            codewords.push_back(read_codeword_up(r, c));
          } else {
            codewords.push_back(read_codeword_down(r, c));
          }
        }
      }
      if(codewords.size() > 3 && codewords[0] == 16){ // Check if type is numeric
        /* // Print Unmasked Code for Verification
        for(int i = 0; i < width; i++){
          for(int j = 0; j < width; j++){
            std::cout << (code_m(i, j) == qr::white ? "\u2588" : " ");
          }
          std::cout << std::endl;
        }
        */
        for(uint8_t i = 0; i <= 0x0f; i++){
          std::vector<uint8_t> working(codewords.size());
          for(int i = 0; i < codewords.size(); i++)
            working[i] = codewords[i];

          working[2] |= ((i) & 1) << 1;
          working[2] |= ((i >> 1) & 1);
          working[3] |= ((i >> 2) & 1) << 7;
          working[3] |= ((i >> 3) & 1) << 6;

          auto extract = [&](int& pos, int len){
            int shift = 24 - (pos & 7) - len;
            int mask = (1 << len) - 1;
            int byteIndex = pos >> 3;
            pos += len;
            return (((working[byteIndex] << 16) | (working[byteIndex + 1] << 8) | working[byteIndex + 2]) >> shift) & mask;
          };

          int bit_idx = 4;
          int n = extract(bit_idx, 10);
          if(n != 4) continue;
          int x = extract(bit_idx, 10);
          uint16_t k = x;
          if(k >= 1000) continue;
          x = extract(bit_idx, 4);
          if(x >= 10) continue;
          dat.push_back(k * 10 + x);
        }
      }
    }
  }
}

std::vector<uint16_t> qr::internal::compute(qr::computed_qr_t bottom){
  std::vector<uint16_t> vals;
  comp(bottom.code().width(), bottom.code(), vals);
  return vals;
}
