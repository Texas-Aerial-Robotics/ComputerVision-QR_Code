#include "qr.h"

inline int qr_value_at_v(int y, int x, int orientation, int width, std::vector<int32_t> data){
  int index = 0;
  if(orientation == 0){ // Top left
    index = (y * (width) + x);
  } else if(orientation == 1){ // Top right
    index = (x * width + (width - y - 1));
  } else if(orientation == 2){ // Bottom right
    index = ((width - y - 1) * (width) + (width - x - 1));
  } else { // Bottom left
    index = ((width - x - 1) * width + y);
  }
  return (data[index / 32] >> (index % 32)) & 1; // unpack bit from qr_portion
}

// The second return value orientation describes the number of 90 degree turns taken for the corner to be at the top left
std::tuple<internal::orientation_t, internal::corner_type_t> internal::check_corner(std::vector<int32_t> d, int width){
  HELPER1(v, width, d)
  HELPER2
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
        orientation = -1;
        break;
      }
      if(orientation == -1) break;
    }
    if(orientation != -1) break;
  }

  if(orientation == -1) return std::make_tuple(static_cast<orientation_t>(0), static_cast<corner_type_t>(2));
  
  bool right = true;
  bool lower = true;
  
  for(int i = 7; i < width; i++){
    if(black(i, 6, orientation, i % 2 != 1)){
      right = false;
    }
    if(black(6, i, orientation, i % 2 != 1)){
      lower = false;
    }
    if(!right && !lower) break;
  }

  return std::make_tuple(static_cast<orientation_t>(orientation), static_cast<corner_type_t>((right && lower) ? 0 : ((right) ? 3 : 1)));
}


