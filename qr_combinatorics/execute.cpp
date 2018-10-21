#include "qr.h"
#include <iostream>

extern void wait_for_cuda_end();

std::string top_left = 
"XXXXXXX   X\n"
"X     X XXX\n"
"X XXX X   X\n"
"X XXX X XX \n"
"X XXX X  X \n"
"X     X X  \n"
"XXXXXXX X X\n"
"         X \n"
"XXXXX XXX  \n"
"XX X   XX X\n"
"XXXXXXX   X";

std::string bottom_right = 
"X X XX  XX \n"
"XXXXX  XXXX\n"
"  X  X  XXX\n"
"  X  X  X  \n"
" X X  X X  \n"
"X    XX XX \n"
"XX X  XXX  \n"
"XXXXXX  X  \n"
"  X XX  X  \n"
" XXXXX   X \n"
"  X  X     \n";

template <int I>
void putstring(std::string test_string, qr_comb_t<I>& t){
  int x_offset = 0, y_offset = 0;
   for(int i = 0; i < test_string.size(); i++){
      if(test_string[i] == ' ')
        x_offset++;
      else if(test_string[i] == 'X'){
        t(x_offset, y_offset) = true;
        x_offset++;
      } else if(test_string[i] == '\n'){
        x_offset = 0;
        y_offset++;
      }
    }
}

int main(){
    qr_comb_t<11> t, b;
    
    putstring(top_left, t);
    putstring(bottom_right, b);
    
    t.compute();
    b.compute();
    
    wait_for_cuda_end();
    return 0;
}
