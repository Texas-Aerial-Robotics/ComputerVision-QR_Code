#include <iostream>
#include <fstream>
using namespace std;

#define top_left 0
#define top_right 1
#define bottom_left 2
#define bottom_right 3

int** change_format(ifstream& myfile) {
		string line;

		getline(myfile, line);

		int length = line.size() / 2;

		// create array to store the qr code
		int** qr_code = new int*[length];
		for(int i = 0; i < length; i++) {
				qr_code[i] = new int[length];
		}

		// populate the array with the qr code
		for(int r = 0; r < length; r++) {
				for(int c = 0; c < length; c++) {
						qr_code[r][c] = line[c*2] == '#' ? 1 : 0;
						//cout << qr_code[r][c] << qr_code[r][c];
				}
				//cout << "\n";

				// get next qr code line
				getline(myfile, line);
		}


		myfile.close();

		return qr_code;

}

int open_file(string filename) {
		int** qr_code;

		string line;
		ifstream myfile(filename);

		if(myfile.is_open()) {
				qr_code = change_format(myfile);
				//cout << qr_code[0][0];
				for (int i = 0; i < 21; i++) {
						for(int j = 0; j < 21; j++) {
								cout << qr_code[i][j];
						}
						cout << "\n";
				}
		} else {
				cout << "Unable to open file\n";
				return 0;
		}
		return 1;
}

int main() {
		open_file("Insert file name here!");
}
