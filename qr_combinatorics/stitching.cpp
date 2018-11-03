#include "qr.h"
#include <cassert>

#define orient_0_x[0] 0
#define orient_0_y 0
#define orient_1_x (width-1)
#define orient_1_y 0
#define orient_2_x (width-1)
#define oreint_2_y (width-1)
#define oreint_3_x 0
#define oreint_3_y (width-1)

#define orient_0_x_dir 1
#define orient_0_y_dir 0
#define orient_1_x_dir 0 
#define orient_1_y_dir 1
#define orient_2_x_dir -1
#define orient_2_y_dir 0
#define orient_3_x_dir 0 
#define orient_3_y_dir -1

void stitching(std::vector<std::tuple<qr_comb_t, int> > qr_quads) {
    using namespace std;

    int size = qr_quads.size();
	assert (size <= 4 && size >= 2);

    int width = get<0>(qr_quads[q1]).getWidth();
    int[][] orient = {{0, 0}, {width - 1, 0}, {width - 1, width - 1}, {0, width - 1}};
    int[][] orient_dir = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};

    for(int q1 = 0; q1 < qr_quads.size() - 1; q1++) {
        for(int q2 = 0; q2 < qr_quads.size(); q2++) {
            // compare q1 with q2 with their given oreintation
            for(int side = 0; side < 4; size++) {
                int x1_temp = orient[(get<1>(qr_quads[q1]) + side) % 4][0];
                int y1_temp = orient[(get<1>(qr_quads[q1]) + side) % 4][1];
                int x2_temp = orient[(get<1>(qr_quads[q1]) + side + 2) % 4][0];
                int y2_temp = orient[(get<1>(qr_quads[q1]) + side + 2 ) % 4][1];

                int x1_dir_temp = orient_dir[(get<1>(qr_quads[q1]) + side) % 4][0];
                int y1_dir_temp = orient_dir[(get<1>(qr_quads[q1]) + side) % 4][1];
                int x2_dir_temp = orient_dir[(get<1>(qr_quads[q1]) + side + 2) % 4][0];
                int y2_dir_temp = orient_dir[(get<1>(qr_quads[q1]) + side + 2 ) % 4][1];

                for(int x1 = x1_temp, y1 = y1_temp, x2 = x2_temp, y2 = y2_temp, counter = 0; counter < width; counter++, x1+=x1_dir_temp, y1+=y1_dir_temp, x2+=x2_dir_temp, y2+=y2_dir_temp) {
                    if(qu_quads(x1, y1) != qr_quads(x2, y2)) {
                        continue;
                    }
                }
                if(side == 0) {
                   get<0>(qr_quads[q1]).unflag(top);
                   get<0>(qr_quads[q2]).flag(top);
                } else if(side == 1) {
                    get<0>(qr_quads[q1]).flag(left);
                    get<0>(qr_quads[q2]).unflag(left);
                } else if(side == 2) {
                   get<0>(qr_quads[q1]).flag(top);
                   get<0>(qr_quads[q2]).unflag(top);
                } else {
                   get<0>(qr_quads[q1]).unflag(left);
                   get<0>(qr_quads[q2]).flag(left);
                }
            }
        }
    }
}
