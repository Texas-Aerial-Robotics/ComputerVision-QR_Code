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
    qr_comb_t<12> t;
    
    putstring(top_left, t);
    
    t.compute();
    
    wait_for_cuda_end();
    return 0;
}
