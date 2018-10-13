#include "qr.h"

inline int qr_value_at_v(int x, int y, int orientation, int width, std::vector<int32_t> data){
  int index = 0;
  if(orientation == 0){ // Top left
    index = (y * (width) + x);
  } else if(orientation == 1){ // Top right
    index = ((width - y - 1) * (width) + x);
  } else if(orientation == 2){ // Bottom left
    index = (y * (width) + (width - x - 1));
  } else { // Bottom right
    index = ((width - y - 1) * (width) + (width - x - 1));
  }
  return (data[index / 32] >> (index % 32)) & 1; // unpack bit from qr_portion
}

// The second return value orientation describes the number of 90 degree turns taken for the corner to be at the top left
std::tuple<int, int> internal::check_corner(std::vector<int32_t> d, int width){
  auto qr_value_at = [&](int x, int y, int orientation){ // Helper function which returns qr code value at a point as 0 or 1.
    return qr_value_at_v(x, y, orientation, width, d);
  };
  auto printerr = [&](int x, int y, int k){ // Helper function which prints qr code with relevant error point
    for(int i = 0; i < width; i++){
      for(int j = 0; j < width; j++){
        if(i == x && j == y) printf("!");
        else printf((qr_value_at(i, j, k) == 1) ? "." : " ");
      }
      printf("\n");
    }
  };
  auto black = [&](int x, int y, int k, bool m = true){
    return qr_value_at(x, y, k) != (m ? 1 : 0);
  };
  int orientation = 0;
  for(int k = 0; k < 4; k++){
    for(int i = 0; i < 7; i++){
      for(int j = 0; j < 7; j++){
        bool expected = (
          (i == 1 && (j != 0 && j != 6)) ||
          (i == 5 && (j != 0 && j != 6)) ||
          (j == 1 && (i != 0 && i != 6)) ||
          (j == 5 && (i != 0 && i != 6))
        );
        orientation = k;
        if(black(i, j, k, expected)) // TODO
          continue;
        printerr(i, j, k);
        orientation = -1;
        break;
      }
      if(orientation != -1) break;
    }
    if(orientation != -1) break;
  }

  if(orientation == -1) return std::make_tuple(0, 2);

  bool right = black(8, 6, orientation) && black(10, 6, orientation);
  bool lower = black(6, 8, orientation) && black(6, 10, orientation);

  return std::make_tuple(orientation, (right && lower) ? 0 : ((right) ? 3 : 1));
}
